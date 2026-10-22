"""Preprocess raw Visual Genome annotations into filtered relationship triplets.

Input:  VRD/vrd_project/data/raw/relationships.json, image_data.json
Output: open_vocab_vrd/data/triplets_train.json
        open_vocab_vrd/data/triplets_val.json
        open_vocab_vrd/data/triplets_test.json
        open_vocab_vrd/data/pred_vocab.json
        open_vocab_vrd/data/obj_vocab.json

Isolation guarantee: all VG format assumptions are confined to
parse_vg_relationships() below. If VG's JSON structure differs, fix only that function.

IMPORTANT: VG subjects use "name" (string) but objects use "names" (list).
Both are handled by _get_vg_name() below.

Usage:
    cd open_vocab_vrd/
    python scripts/preprocess_vg.py --config configs/default.yaml
    python scripts/preprocess_vg.py --config configs/default.yaml --max_images 500
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
    clamp_box,
    filter_objects,
    filter_predicates,
    split_by_image,
    xywh_to_xyxy,
)
from src.utils.io import load_yaml, save_json


# --------------------------------------------------------------------------- #
# VG format parser
# VG raw JSON (confirmed against VG 1.4 on this machine):
#
#   subject dict: {"name": "shade", "x":..., "y":..., "w":..., "h":...}  <- str
#   object  dict: {"names": ["sidewalk"], "x":..., ...}                   <- list
#
# Both cases handled by _get_vg_name(). Fix only here if format differs.
# --------------------------------------------------------------------------- #


def _get_vg_name(d: dict) -> str:
    """Extract a clean name from a VG subject or object dict.

    Handles both:
      "name": "person"       (subject format)
      "names": ["sidewalk"]  (object format)
    """
    name = d.get("name") or ""
    if not name:
        names = d.get("names") or []
        name = names[0] if names else ""
    return name.strip().lower()


def parse_vg_relationships(
    rel_path: Path,
    img_meta_path: Path,
    verbose: bool = True,
    max_images: int = None,
) -> list:
    """Parse raw VG relationships.json into a flat list of triplet dicts.

    Returns list of:
    {
        "image_id": int,
        "image_file": "12345.jpg",
        "image_w": int,
        "image_h": int,
        "subject_name": str,
        "predicate": str (lowercased),
        "object_name": str,
        "subject_box": [x1, y1, x2, y2],
        "object_box":  [x1, y1, x2, y2],
    }
    """
    print("Loading image metadata...")
    with open(img_meta_path, "r", encoding="utf-8") as f:
        img_data = json.load(f)
    img_meta = {d["image_id"]: d for d in img_data}

    print("Loading relationships...")
    with open(rel_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    triplets = []
    skipped_no_meta = 0
    skipped_bad_box = 0
    skipped_no_name = 0

    for img_idx, img_entry in enumerate(raw):
        if max_images is not None and img_idx >= max_images:
            break
        img_id = img_entry["image_id"]
        rels   = img_entry.get("relationships", [])


        meta = img_meta.get(img_id)
        if meta is None:
            skipped_no_meta += len(rels)
            continue

        img_w = float(meta.get("width",  1))
        img_h = float(meta.get("height", 1))
        img_file = f"{img_id}.jpg"

        for rel in rels:
            subj = rel.get("subject", {})
            obj  = rel.get("object",  {})
            pred = rel.get("predicate", "").strip().lower()

            subj_name = _get_vg_name(subj)
            obj_name  = _get_vg_name(obj)

            if not subj_name or not obj_name or not pred:
                skipped_no_name += 1
                continue

            # Convert [x, y, w, h] -> [x1, y1, x2, y2] and clamp
            sx1, sy1, sx2, sy2 = xywh_to_xyxy(
                subj.get("x", 0), subj.get("y", 0),
                subj.get("w", 0), subj.get("h", 0)
            )
            ox1, oy1, ox2, oy2 = xywh_to_xyxy(
                obj.get("x", 0), obj.get("y", 0),
                obj.get("w", 0), obj.get("h", 0)
            )

            sx1, sy1, sx2, sy2 = clamp_box(sx1, sy1, sx2, sy2, img_w, img_h)
            ox1, oy1, ox2, oy2 = clamp_box(ox1, oy1, ox2, oy2, img_w, img_h)

            # Skip degenerate boxes
            if sx2 - sx1 < 2 or sy2 - sy1 < 2:
                skipped_bad_box += 1
                continue
            if ox2 - ox1 < 2 or oy2 - oy1 < 2:
                skipped_bad_box += 1
                continue

            triplets.append({
                "image_id":    img_id,
                "image_file":  img_file,
                "image_w":     int(img_w),
                "image_h":     int(img_h),
                "subject_name": subj_name,
                "predicate":    pred,
                "object_name":  obj_name,
                "subject_box":  [int(sx1), int(sy1), int(sx2), int(sy2)],
                "object_box":   [int(ox1), int(oy1), int(ox2), int(oy2)],
            })

    if verbose:
        print(f"  Parsed {len(triplets):,} valid triplets.")
        print(f"  Skipped - no image meta: {skipped_no_meta}, bad box: {skipped_bad_box}, no name: {skipped_no_name}")
    return triplets


def main():
    parser = argparse.ArgumentParser(description="Preprocess VG relationships into triplet JSON files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Stop parsing after this many images (fast debug mode).")
    parser.add_argument("--max_triplets", type=int, default=None,
                        help="Cap total triplets after filtering.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg  = cfg["data"]
    raw_dir   = Path(data_cfg["vg_raw_dir"])
    out_dir   = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load vocab filters from predicates.yaml (absolute path — works from any CWD)
    pred_cfg_path = Path(__file__).parent.parent / "configs" / "predicates.yaml"
    pred_cfg = load_yaml(pred_cfg_path)
    allowed_predicates = set(pred_cfg["predicates"])
    allowed_objects    = set(pred_cfg["objects"])

    print(f"Allowed predicates ({len(allowed_predicates)}): {sorted(allowed_predicates)}")
    print(f"Allowed objects    ({len(allowed_objects)}):    (first 10) {sorted(allowed_objects)[:10]} ...")

    # Parse raw VG
    rel_path      = raw_dir / "relationships.json"
    img_meta_path = raw_dir / "image_data.json"

    if not rel_path.exists():
        print(f"ERROR: {rel_path} not found.")
        print("Download Visual Genome from https://visualgenome.org/api/v0/api_home.html")
        sys.exit(1)

    triplets = parse_vg_relationships(rel_path, img_meta_path, max_images=args.max_images)

    # Filter
    triplets = filter_predicates(triplets, allowed_predicates)
    print(f"After predicate filter: {len(triplets):,}")

    triplets = filter_objects(triplets, allowed_objects)
    print(f"After object filter:    {len(triplets):,}")

    if args.max_triplets:
        triplets = triplets[: args.max_triplets]
        print(f"Capped to {len(triplets):,} triplets (--max_triplets).")

    if not triplets:
        print("ERROR: No triplets remain after filtering. Check predicates.yaml and raw data paths.")
        sys.exit(1)

    # Build vocabularies
    pred_vocab = build_predicate_vocab(triplets)
    obj_vocab  = build_object_vocab(triplets)
    print(f"Predicate vocab size: {len(pred_vocab)}")
    print(f"Object vocab size:    {len(obj_vocab)}")

    # Split
    train, val, test = split_by_image(triplets)
    print(f"Split - train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    # Save
    save_json(train,      out_dir / "triplets_train.json")
    save_json(val,        out_dir / "triplets_val.json")
    save_json(test,       out_dir / "triplets_test.json")
    save_json(pred_vocab, out_dir / "pred_vocab.json")
    save_json(obj_vocab,  out_dir / "obj_vocab.json")

    print(f"\nDone. Files written to {out_dir}/")
    print("  triplets_train.json / _val.json / _test.json")
    print("  pred_vocab.json / obj_vocab.json")


if __name__ == "__main__":
    main()
