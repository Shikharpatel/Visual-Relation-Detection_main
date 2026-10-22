"""
COCO-to-VG Class Mapping
=========================

Maps COCO detection class indices to Visual Genome object class indices.
Faster R-CNN (pretrained on COCO) outputs 91 COCO class IDs, while the
VG dataset uses 104 object classes. This module provides the mapping.

Since COCO has coarse labels (e.g., "person") while VG has fine-grained
labels (e.g., "man", "woman", "boy", "girl"), we provide a multi-mapping
that returns ALL possible VG labels for each COCO detection. This enables
generating multiple prediction candidates for evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# COCO class names (index -> name), matching torchvision's pretrained Faster R-CNN
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO name -> list of VG names to try (ordered by likelihood)
# This is the multi-mapping: each COCO class can correspond to multiple VG classes
_COCO_TO_VG_ALIASES = {
    'person': ['man', 'person', 'woman', 'people', 'boy', 'girl'],
    'car': ['car'],
    'bus': ['bus'],
    'train': ['train'],
    'truck': ['car'],
    'boat': ['boat'],
    'bench': ['bench'],
    'bird': ['bird'],
    'umbrella': ['umbrella'],
    'bottle': ['bottle'],
    'bowl': ['bowl'],
    'chair': ['chair'],
    'clock': ['clock'],
    'handbag': ['bag'],
    'bicycle': ['wheel'],
    'couch': ['chair'],
    'dining table': ['table'],
    'potted plant': ['flower', 'flowers'],
    'bed': ['table'],
    'tv': ['glass', 'mirror'],
    'laptop': ['box'],
    'cup': ['glass', 'bowl'],
    'vase': ['bottle'],
    'book': ['box'],
    'suitcase': ['bag'],
}


def build_coco_to_vg_mapping(
    obj_classes_path: str,
    default_class: int = 0
) -> Dict[int, int]:
    """
    Build a single-label mapping from COCO class indices to VG class indices.
    Returns the first (most likely) VG class for each COCO class.

    Args:
        obj_classes_path: Path to VG obj_classes.json (name -> index)
        default_class: VG class index to use for unmapped COCO classes

    Returns:
        Dict mapping COCO class index -> VG class index
    """
    multi = build_coco_to_vg_multi_mapping(obj_classes_path, default_class)
    return {k: v[0] for k, v in multi.items()}


def build_coco_to_vg_multi_mapping(
    obj_classes_path: str,
    default_class: int = 0
) -> Dict[int, List[int]]:
    """
    Build a multi-label mapping from COCO class indices to VG class indices.
    Each COCO class maps to a list of possible VG classes.

    Args:
        obj_classes_path: Path to VG obj_classes.json (name -> index)
        default_class: VG class index to use for unmapped COCO classes

    Returns:
        Dict mapping COCO class index -> list of VG class indices
    """
    with open(obj_classes_path, 'r') as f:
        vg_classes = json.load(f)  # name -> index

    # Build lowercase lookup
    vg_lower = {name.lower(): idx for name, idx in vg_classes.items()}

    mapping = {}

    for coco_idx, coco_name in enumerate(COCO_CLASSES):
        if coco_name == '__background__' or coco_name == 'N/A':
            mapping[coco_idx] = [0]
            continue

        candidates = []

        # Try alias mapping first (has explicit priority ordering)
        if coco_name in _COCO_TO_VG_ALIASES:
            for alias in _COCO_TO_VG_ALIASES[coco_name]:
                if alias.lower() in vg_lower:
                    candidates.append(vg_lower[alias.lower()])

        # Try exact match if not already found
        coco_lower = coco_name.lower()
        if coco_lower in vg_lower:
            exact_idx = vg_lower[coco_lower]
            if exact_idx not in candidates:
                candidates.insert(0, exact_idx)

        if candidates:
            mapping[coco_idx] = candidates
        else:
            mapping[coco_idx] = [default_class]

    return mapping


def load_coco_to_vg_mapping(
    data_dir: str,
    default_class: int = 0
) -> Dict[int, int]:
    """
    Convenience function to load single-label COCO-to-VG mapping.

    Args:
        data_dir: Path to processed data directory containing obj_classes.json
        default_class: VG class index for unmapped COCO classes

    Returns:
        Dict mapping COCO class index -> VG class index
    """
    obj_classes_path = Path(data_dir) / 'obj_classes.json'
    if not obj_classes_path.exists():
        raise FileNotFoundError(f"obj_classes.json not found in {data_dir}")
    return build_coco_to_vg_mapping(str(obj_classes_path), default_class)


def load_coco_to_vg_multi_mapping(
    data_dir: str,
    default_class: int = 0
) -> Dict[int, List[int]]:
    """
    Convenience function to load multi-label COCO-to-VG mapping.

    Args:
        data_dir: Path to processed data directory containing obj_classes.json
        default_class: VG class index for unmapped COCO classes

    Returns:
        Dict mapping COCO class index -> list of VG class indices
    """
    obj_classes_path = Path(data_dir) / 'obj_classes.json'
    if not obj_classes_path.exists():
        raise FileNotFoundError(f"obj_classes.json not found in {data_dir}")
    return build_coco_to_vg_multi_mapping(str(obj_classes_path), default_class)
