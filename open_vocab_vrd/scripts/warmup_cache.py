"""Pre-compute and cache CLIP embeddings for all training triplets.

Run this ONCE before training. After warmup, every training epoch reads
embeddings from disk (microseconds per sample) instead of running CLIP
(tens of milliseconds per crop on GPU).

This script must use num_workers=0 (CLIP is on CUDA, cannot be forked).

Usage:
    cd open_vocab_vrd/
    python scripts/warmup_cache.py --config configs/default.yaml
    python scripts/warmup_cache.py --config configs/debug.yaml --splits train val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.encoders.clip_encoder import CLIPEncoder, make_clip_cache_key
from src.utils.cache import DiskCache
from src.utils.io import load_yaml


def warmup_split(
    triplets_path: Path,
    images_dir: Path,
    clip_encoder: CLIPEncoder,
    clip_cache: DiskCache,
) -> dict:
    """Iterate triplets, compute and cache CLIP embeddings. Returns stats."""
    with open(triplets_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    hits = 0
    computed = 0
    missing_img = 0
    errors = 0

    # Group by image to batch crops per image efficiently
    from collections import defaultdict
    by_image: dict = defaultdict(list)
    for idx, t in enumerate(triplets):
        by_image[t["image_id"]].append((idx, t))

    for img_id, group in tqdm(by_image.items(), desc=f"  {triplets_path.name}"):
        img_file = group[0][1]["image_file"]
        img_path = images_dir / img_file
        if not img_path.exists():
            missing_img += len(group)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            errors += len(group)
            continue

        # Collect all unique boxes in this image across all triplets
        # Cache is keyed per (subject_box, object_box) pair, so we compute per triplet
        for idx, t in group:
            boxes = [t["subject_box"], t["object_box"]]
            key = make_clip_cache_key(img_id, boxes)

            if clip_cache.exists(key):
                hits += 1
                continue

            try:
                embs = clip_encoder.encode_crops(image, boxes)  # (2, 512)
                clip_cache.save(key, embs)
                computed += 1
            except Exception as e:
                errors += 1

    return {"hits": hits, "computed": computed, "missing_img": missing_img, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to warm up.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config using absolute path so this works from any CWD
    cfg_path = Path(__file__).parent.parent / args.config
    cfg = load_yaml(cfg_path)
    data_cfg  = cfg["data"]
    clip_cfg  = cfg["clip"]
    cache_cfg = cfg["cache"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve paths relative to the project root (open_vocab_vrd/)
    project_root = Path(__file__).parent.parent
    images_dir   = (project_root / data_cfg["images_dir"]).resolve()
    processed_dir = project_root / data_cfg["processed_dir"]
    cache_root   = project_root / data_cfg["cache_dir"]

    if not images_dir.exists():
        print(f"ERROR: images_dir not found: {images_dir}")
        sys.exit(1)

    clip_cache = DiskCache(
        cache_root / cache_cfg.get("clip_subdir", "clip_embeddings"),
        enabled=True,
    )

    print(f"Loading CLIP {clip_cfg.get('model', 'ViT-B/32')}...")
    clip_encoder = CLIPEncoder(
        model_name=clip_cfg.get("model", "ViT-B/32"),
        device=device,
        cache=None,  # don't double-cache inside encoder; we cache here
        batch_size=clip_cfg.get("batch_size", 32),
    )

    total_stats: dict = {"hits": 0, "computed": 0, "missing_img": 0, "errors": 0}

    for split in args.splits:
        triplets_path = processed_dir / f"triplets_{split}.json"
        if not triplets_path.exists():
            print(f"WARNING: {triplets_path} not found, skipping.")
            continue
        print(f"\nWarming up {split} split...")
        stats = warmup_split(triplets_path, images_dir, clip_encoder, clip_cache)
        print(f"  Cache hits={stats['hits']}  Computed={stats['computed']}  "
              f"Missing images={stats['missing_img']}  Errors={stats['errors']}")
        for k in total_stats:
            total_stats[k] += stats[k]

    total_seen = total_stats["hits"] + total_stats["computed"] + total_stats["missing_img"] + total_stats["errors"]
    total_cached = total_stats["hits"] + total_stats["computed"]
    hit_rate = total_stats["hits"] / max(total_cached, 1) * 100

    print(f"\n{'='*50}")
    print(f"Warmup complete.")
    print(f"  Total triplets seen:  {total_seen:,}")
    print(f"  Already cached (hit): {total_stats['hits']:,}  ({hit_rate:.1f}% of reachable)")
    print(f"  Newly computed:       {total_stats['computed']:,}")
    print(f"  Missing images:       {total_stats['missing_img']:,}  (VG images not on disk)")
    print(f"  Errors:               {total_stats['errors']:,}")
    print(f"  Coverage:             {total_cached:,}/{total_seen:,} "
          f"({total_cached / max(total_seen, 1) * 100:.1f}%)")
    print(f"\nCache at: {cache_root / cache_cfg.get('clip_subdir', 'clip_embeddings')}")
    if total_stats["missing_img"] > 0:
        print(f"NOTE: {total_stats['missing_img']:,} triplets have no local image.")
        print("      They will be excluded from training (no random embeddings).")
    print("Training will now be fast (disk reads only, no CLIP inference).")


if __name__ == "__main__":
    main()
