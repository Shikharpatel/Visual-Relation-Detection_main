"""Utility functions used by preprocess_vg.py and vg_dataset.py.

Isolated here so any VG format quirks are in one place and easy to fix.
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Box helpers
# --------------------------------------------------------------------------- #

def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert VG-style [x, y, w, h] to [x1, y1, x2, y2]."""
    return float(x), float(y), float(x + w), float(y + h)


def is_valid_box(x1: float, y1: float, x2: float, y2: float,
                 img_w: float, img_h: float,
                 min_size: float = 2.0) -> bool:
    """Return True if the box has positive area and fits inside the image."""
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return False
    if x1 < 0 or y1 < 0:
        return False
    if x2 > img_w or y2 > img_h:
        return False
    return True


def clamp_box(x1: float, y1: float, x2: float, y2: float,
              img_w: float, img_h: float) -> Tuple[float, float, float, float]:
    """Clamp box coords to image boundaries."""
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    return x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# Triplet filtering
# --------------------------------------------------------------------------- #

def filter_predicates(triplets: List[Dict], allowed: Set[str]) -> List[Dict]:
    """Keep only triplets whose predicate is in the allowed set (case-insensitive)."""
    allowed_lower = {p.lower() for p in allowed}
    return [t for t in triplets if t["predicate"].lower() in allowed_lower]


def filter_objects(triplets: List[Dict], allowed: Set[str]) -> List[Dict]:
    """Keep only triplets where both subject and object are in the allowed set."""
    allowed_lower = {o.lower() for o in allowed}
    return [
        t for t in triplets
        if t["subject_name"].lower() in allowed_lower
        and t["object_name"].lower() in allowed_lower
    ]


def build_predicate_vocab(triplets: List[Dict]) -> List[str]:
    """Return a sorted list of unique predicates present in the triplets."""
    return sorted({t["predicate"].lower() for t in triplets})


def build_object_vocab(triplets: List[Dict]) -> List[str]:
    """Return sorted list of unique object/subject names."""
    names = set()
    for t in triplets:
        names.add(t["subject_name"].lower())
        names.add(t["object_name"].lower())
    return sorted(names)


# --------------------------------------------------------------------------- #
# Train / val / test split (stable, by image_id)
# --------------------------------------------------------------------------- #

def split_by_image(
    triplets: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split triplets into train/val/test partitions by image_id.

    Images (not triplets) are shuffled so all triplets from one image
    land in the same split, preventing leakage.
    """
    rng = np.random.default_rng(seed)
    image_ids = sorted({t["image_id"] for t in triplets})
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = set(image_ids[:n_train])
    val_ids   = set(image_ids[n_train : n_train + n_val])
    test_ids  = set(image_ids[n_train + n_val :])

    train = [t for t in triplets if t["image_id"] in train_ids]
    val   = [t for t in triplets if t["image_id"] in val_ids]
    test  = [t for t in triplets if t["image_id"] in test_ids]

    return train, val, test


# --------------------------------------------------------------------------- #
# Predicate class-frequency stats
# --------------------------------------------------------------------------- #

def predicate_frequencies(triplets: List[Dict]) -> Dict[str, int]:
    """Count occurrences of each predicate in the triplet list."""
    return dict(Counter(t["predicate"].lower() for t in triplets))


def top_k_predicates(triplets: List[Dict], k: int) -> List[str]:
    """Return the k most frequent predicates."""
    freq = predicate_frequencies(triplets)
    return [p for p, _ in sorted(freq.items(), key=lambda x: -x[1])[:k]]
