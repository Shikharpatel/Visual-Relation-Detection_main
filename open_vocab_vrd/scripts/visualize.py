"""Visualize VRD predictions on an image.

Reads predictions from the JSON produced by scripts/infer.py and draws:
  - Bounding boxes per detected object
  - Object labels
  - Arrows between related pairs with predicate + confidence

Usage:
    cd open_vocab_vrd/
    python scripts/visualize.py \\
        --image path/to/image.jpg \\
        --predictions predictions.json \\
        --output vis.png \\
        --top_k 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.utils.visualization import draw_scene_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True, help="Path to input image.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON (from infer.py).")
    parser.add_argument("--output",      default="vis.png", help="Output image path.")
    parser.add_argument("--top_k",       type=int, default=10, help="Max triplets to draw.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    preds_path = Path(args.predictions)
    if not preds_path.exists():
        print(f"ERROR: predictions not found: {preds_path}")
        sys.exit(1)

    with open(preds_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    print(f"Loaded {len(triplets)} triplets from {preds_path}")

    image = Image.open(image_path).convert("RGB")
    out_img = draw_scene_graph(image, triplets, max_triplets=args.top_k, save_path=args.output)

    print(f"Saved visualization to {args.output}")
    print(f"Image size: {out_img.size}")


if __name__ == "__main__":
    main()
