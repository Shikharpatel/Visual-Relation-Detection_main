"""Fast preprocessor: converts vrd_project/data/processed/ -> open_vocab_vrd/data/triplets.

This is the PREFERRED alternative to preprocess_vg.py.
It reads the already-processed vrd_project data (38 MB instead of 700 MB raw),
which is ~20× faster and avoids all raw VG format quirks.

vrd_project processed format (per image entry):
  {
    "image_id": int, "width": int, "height": int,
    "objects": [{"idx": int, "name": str, "class_idx": int, "box": [x1,y1,x2,y2]}, ...],
    "relations": [{"subject_idx": int, "object_idx": int, "predicate": str, "predicate_idx": int}, ...]
  }

Output triplet format (open_vocab_vrd):
  {
    "image_id": int, "image_file": "12345.jpg",
    "image_w": int, "image_h": int,
    "subject_name": str, "predicate": str, "object_name": str,
    "subject_box": [x1,y1,x2,y2], "object_box": [x1,y1,x2,y2]
  }

Usage:
    cd open_vocab_vrd/
    python scripts/preprocess_from_processed.py --config configs/default.yaml
    python scripts/preprocess_from_processed.py --config configs/debug.yaml --max_images 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.preprocess_utils import (
    build_object_vocab,
    build_predicate_vocab,
    filter_objects,
    filter_predicates,
    split_by_image,
)
from src.utils.io import load_yaml, save_json

# Path to vrd_project processed splits (relative to open_vocab_vrd/)
VRD_PROCESSED = Path(__file__).parent.parent.parent / "vrd_project" / "data" / "processed"


def convert_split(split_path: Path, allowed_predicates: set, allowed_objects: set,
                  max_images: int = None) -> list:
    """Read one vrd_project split and convert to open_vocab_vrd triplet format."""
    print(f"Reading {split_path.name}...")
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triplets = []
    skipped_no_box = 0
    skipped_no_pred = 0

    for img_idx, entry in enumerate(data):
        if max_images is not None and img_idx >= max_images:
            break

        img_id = entry["image_id"]
        img_w  = entry.get("width", 1)
        img_h  = entry.get("height", 1)
        img_file = f"{img_id}.jpg"

        objects = entry.get("objects", [])
        # Build idx -> object dict for fast lookup
        obj_by_idx = {o["idx"]: o for o in objects}

        for rel in entry.get("relations", []):
            pred = rel.get("predicate", "").strip().lower()
            if not pred:
                skipped_no_pred += 1
                continue

            subj_dict = obj_by_idx.get(rel["subject_idx"])
            obj_dict  = obj_by_idx.get(rel["object_idx"])
            if subj_dict is None or obj_dict is None:
                skipped_no_box += 1
                continue

            subj_name = subj_dict.get("name", "").strip().lower()
            obj_name  = obj_dict.get("name", "").strip().lower()
            if not subj_name or not obj_name:
                skipped_no_box += 1
                continue

            subj_box = subj_dict.get("box")
            obj_box  = obj_dict.get("box")
            if subj_box is None or obj_box is None:
                skipped_no_box += 1
                continue

            # Sanity: boxes should be [x1,y1,x2,y2] with positive area
            if subj_box[2] - subj_box[0] < 2 or subj_box[3] - subj_box[1] < 2:
                skipped_no_box += 1
                continue
            if obj_box[2] - obj_box[0] < 2 or obj_box[3] - obj_box[1] < 2:
                skipped_no_box += 1
                continue

            triplets.append({
                "image_id":    img_id,
                "image_file":  img_file,
                "image_w":     img_w,
                "image_h":     img_h,
                "subject_name": subj_name,
                "predicate":    pred,
                "object_name":  obj_name,
                "subject_box":  subj_box,
                "object_box":   obj_box,
            })

    if allowed_predicates:
        triplets = filter_predicates(triplets, allowed_predicates)
    if allowed_objects:
        triplets = filter_objects(triplets, allowed_objects)

    print(f"  -> {len(triplets):,} triplets  "
          f"(skipped bad-box: {skipped_no_box}, no-pred: {skipped_no_pred})")
    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Convert vrd_project processed splits to open_vocab_vrd triplet format."
    )
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit images PER SPLIT (debug mode).")
    parser.add_argument("--no_filter",  action="store_true",
                        help="Skip predicate/object filtering (use all 53 predicates).")
    args = parser.parse_args()

    cfg     = load_yaml(Path(__file__).parent.parent / args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_cfg_path = Path(__file__).parent.parent / "configs" / "predicates.yaml"
    pred_cfg = load_yaml(pred_cfg_path)

    if args.no_filter:
        allowed_predicates = set()
        allowed_objects    = set()
        print("No predicate/object filtering - using all VG classes.")
    else:
        allowed_predicates = set(p.lower() for p in pred_cfg["predicates"])
        allowed_objects    = set(o.lower() for o in pred_cfg["objects"])
        print(f"Allowed predicates ({len(allowed_predicates)}): {sorted(allowed_predicates)}")

    if not VRD_PROCESSED.exists():
        print(f"ERROR: {VRD_PROCESSED} not found.")
        print("Make sure vrd_project/data/processed/ exists (it was built by the existing VRD pipeline).")
        sys.exit(1)

    # Read each split from vrd_project processed files
    all_triplets = []
    for split_name in ("train", "val", "test"):
        split_path = VRD_PROCESSED / f"relationships_{split_name}.json"
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping.")
            continue
        t = convert_split(split_path, allowed_predicates, allowed_objects, args.max_images)
        all_triplets.append((split_name, t))

    if not all_triplets:
        print("ERROR: No triplets loaded from any split.")
        sys.exit(1)

    # Save each split directly (already split by vrd_project)
    pred_vocab_all = []
    for split_name, triplets in all_triplets:
        save_json(triplets, out_dir / f"triplets_{split_name}.json")
        print(f"Saved {split_name}: {len(triplets):,} triplets")
        pred_vocab_all.extend(t["predicate"] for t in triplets)

    # Build pred vocab from training split (or all if train missing)
    train_triplets = next((t for n, t in all_triplets if n == "train"), all_triplets[0][1])
    pred_vocab = build_predicate_vocab(train_triplets)
    obj_vocab  = build_object_vocab(train_triplets)

    save_json(pred_vocab, out_dir / "pred_vocab.json")
    save_json(obj_vocab,  out_dir / "obj_vocab.json")

    print(f"\nPredicate vocab ({len(pred_vocab)}): {pred_vocab}")
    print(f"Object vocab size: {len(obj_vocab)}")
    print(f"\nDone. Files in {out_dir}/")
    print("  triplets_train/val/test.json  pred_vocab.json  obj_vocab.json")
    print("\nNext: python scripts/build_llm_prior_stub.py --config", args.config)


if __name__ == "__main__":
    main()
