"""Draw detected objects and relationship triplets on an image.

Output: a PIL image or saved PNG with:
  - Colored bounding boxes per object
  - Object label text
  - Arrows between related object pairs
  - Relationship label along the arrow
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Consistent color per object index (cycles through 20 colors)
_COLORS = [
    (255, 87, 87), (87, 255, 87), (87, 87, 255), (255, 215, 0),
    (0, 255, 215), (215, 0, 255), (255, 140, 0), (0, 191, 255),
    (255, 20, 147), (50, 205, 50), (255, 165, 0), (138, 43, 226),
    (0, 250, 154), (255, 69, 0), (30, 144, 255), (255, 228, 181),
    (0, 139, 139), (220, 20, 60), (107, 142, 35), (255, 182, 193),
]


def _box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_scene_graph(
    image: Image.Image,
    triplets: List[Dict],
    objects: Optional[List[Dict]] = None,
    max_triplets: int = 15,
    save_path: Optional[str | Path] = None,
) -> Image.Image:
    """Render VRD predictions on an image.

    Args:
        image: PIL Image (RGB).
        triplets: List of dicts with keys:
            subject, predicate, object, confidence,
            subject_box [x1,y1,x2,y2], object_box [x1,y1,x2,y2].
        objects: Optional list of {label, box, score} for all detections.
            If None, boxes are inferred from triplets.
        max_triplets: Only draw top-N by confidence.
        save_path: If given, save output image here.

    Returns:
        PIL Image with annotations.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to load a small font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        small_font = font

    # Collect all unique boxes and assign color indices
    box_color: Dict[tuple, Tuple[int, int, int]] = {}
    box_label: Dict[tuple, str] = {}
    color_idx = 0

    top_trips = sorted(triplets, key=lambda x: x.get("confidence", 0), reverse=True)[:max_triplets]

    for t in top_trips:
        for side in ("subject", "object"):
            box_key = tuple(map(int, t[f"{side}_box"]))
            if box_key not in box_color:
                box_color[box_key] = _COLORS[color_idx % len(_COLORS)]
                box_label[box_key] = t[side]
                color_idx += 1

    # Draw boxes
    for box_key, color in box_color.items():
        x1, y1, x2, y2 = box_key
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = box_label[box_key]
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill=(0, 0, 0), font=font)

    # Draw relation arrows
    for t in top_trips:
        sx, sy = _box_center(t["subject_box"])
        ox, oy = _box_center(t["object_box"])
        # Semi-transparent line
        draw.line([(sx, sy), (ox, oy)], fill=(255, 255, 255, 180), width=2)
        # Label at midpoint
        mx, my = (sx + ox) / 2, (sy + oy) / 2
        rel_label = f"{t['predicate']} ({t['confidence']:.2f})"
        draw.text((mx, my), rel_label, fill=(255, 255, 0), font=small_font)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img
