"""
Bounding Box Operations for VRD
================================

Provides utilities for box IoU, union computation, and spatial feature encoding.
All boxes are in format [x1, y1, x2, y2] (top-left, bottom-right).
"""

import numpy as np
import torch
from typing import Tuple, Union, Optional

ArrayLike = Union[np.ndarray, torch.Tensor]


def box_area(boxes: ArrayLike) -> ArrayLike:
    """
    Compute area of boxes.

    Args:
        boxes: [..., 4] array of boxes [x1, y1, x2, y2]

    Returns:
        [...] array of areas
    """
    if isinstance(boxes, torch.Tensor):
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: ArrayLike, boxes2: ArrayLike) -> ArrayLike:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes
        boxes2: [M, 4] boxes

    Returns:
        [N, M] IoU matrix
    """
    if isinstance(boxes1, torch.Tensor):
        return _box_iou_torch(boxes1, boxes2)
    return _box_iou_numpy(boxes1, boxes2)


def _box_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of box IoU."""
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    # IoU
    iou = inter / union.clamp(min=1e-8)
    return iou


def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """NumPy implementation of box IoU."""
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-8)

    # IoU
    iou = inter / union
    return iou


def box_iou_pairwise(boxes1: ArrayLike, boxes2: ArrayLike) -> ArrayLike:
    """
    Compute IoU for corresponding pairs of boxes.

    Args:
        boxes1: [N, 4] boxes
        boxes2: [N, 4] boxes

    Returns:
        [N] IoU values for each pair
    """
    if isinstance(boxes1, torch.Tensor):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-8)
    else:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = np.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, 2:], boxes2[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, 0] * wh[:, 1]

        union = np.maximum(area1 + area2 - inter, 1e-8)
        return inter / union


def box_union(boxes1: ArrayLike, boxes2: ArrayLike) -> ArrayLike:
    """
    Compute union (enclosing) box for pairs of boxes.

    Args:
        boxes1: [N, 4] boxes
        boxes2: [N, 4] boxes (or [M, 4] for broadcast)

    Returns:
        [N, 4] (or [N, M, 4]) union boxes
    """
    if isinstance(boxes1, torch.Tensor):
        if boxes1.dim() == 2 and boxes2.dim() == 2 and boxes1.size(0) == boxes2.size(0):
            # Pairwise
            x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
            y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
            x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
            y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
            return torch.stack([x1, y1, x2, y2], dim=1)
        else:
            # Broadcast
            x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
            y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
            x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
            y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
            return torch.stack([x1, y1, x2, y2], dim=2)
    else:
        if boxes1.ndim == 2 and boxes2.ndim == 2 and boxes1.shape[0] == boxes2.shape[0]:
            x1 = np.minimum(boxes1[:, 0], boxes2[:, 0])
            y1 = np.minimum(boxes1[:, 1], boxes2[:, 1])
            x2 = np.maximum(boxes1[:, 2], boxes2[:, 2])
            y2 = np.maximum(boxes1[:, 3], boxes2[:, 3])
            return np.stack([x1, y1, x2, y2], axis=1)
        else:
            x1 = np.minimum(boxes1[:, None, 0], boxes2[None, :, 0])
            y1 = np.minimum(boxes1[:, None, 1], boxes2[None, :, 1])
            x2 = np.maximum(boxes1[:, None, 2], boxes2[None, :, 2])
            y2 = np.maximum(boxes1[:, None, 3], boxes2[None, :, 3])
            return np.stack([x1, y1, x2, y2], axis=2)


def normalize_boxes(
    boxes: ArrayLike,
    image_size: Tuple[int, int]
) -> ArrayLike:
    """
    Normalize box coordinates to [0, 1] range.

    Args:
        boxes: [..., 4] boxes in pixel coordinates
        image_size: (height, width) of image

    Returns:
        [..., 4] normalized boxes
    """
    h, w = image_size

    if isinstance(boxes, torch.Tensor):
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        return boxes / scale
    else:
        scale = np.array([w, h, w, h])
        return boxes / scale


def compute_spatial_features(
    subj_boxes: ArrayLike,
    obj_boxes: ArrayLike,
    image_size: Optional[Tuple[int, int]] = None
) -> ArrayLike:
    """
    Compute spatial features for subject-object box pairs.

    Features computed:
    1. dx: (obj_cx - subj_cx) / subj_w  (relative x displacement)
    2. dy: (obj_cy - subj_cy) / subj_h  (relative y displacement)
    3. dw: log(obj_w / subj_w)          (relative width)
    4. dh: log(obj_h / subj_h)          (relative height)
    5. IoU: intersection over union
    6-9. Normalized subject box [x1, y1, x2, y2] / image_size
    10-13. Normalized object box [x1, y1, x2, y2] / image_size

    If image_size is provided, returns 13 features.
    Otherwise, returns 5 features.

    Args:
        subj_boxes: [N, 4] subject boxes
        obj_boxes: [N, 4] object boxes
        image_size: Optional (height, width) for normalization

    Returns:
        [N, 5] or [N, 13] spatial features
    """
    if isinstance(subj_boxes, torch.Tensor):
        return _compute_spatial_features_torch(subj_boxes, obj_boxes, image_size)
    return _compute_spatial_features_numpy(subj_boxes, obj_boxes, image_size)


def _compute_spatial_features_torch(
    subj_boxes: torch.Tensor,
    obj_boxes: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """PyTorch implementation of spatial features."""
    # Box dimensions
    subj_w = subj_boxes[:, 2] - subj_boxes[:, 0]
    subj_h = subj_boxes[:, 3] - subj_boxes[:, 1]
    obj_w = obj_boxes[:, 2] - obj_boxes[:, 0]
    obj_h = obj_boxes[:, 3] - obj_boxes[:, 1]

    # Centers
    subj_cx = (subj_boxes[:, 0] + subj_boxes[:, 2]) / 2
    subj_cy = (subj_boxes[:, 1] + subj_boxes[:, 3]) / 2
    obj_cx = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2
    obj_cy = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2

    # Relative displacement
    dx = (obj_cx - subj_cx) / subj_w.clamp(min=1e-8)
    dy = (obj_cy - subj_cy) / subj_h.clamp(min=1e-8)

    # Relative size (log scale)
    dw = torch.log((obj_w / subj_w.clamp(min=1e-8)).clamp(min=1e-8))
    dh = torch.log((obj_h / subj_h.clamp(min=1e-8)).clamp(min=1e-8))

    # IoU
    iou = box_iou_pairwise(subj_boxes, obj_boxes)

    features = [dx, dy, dw, dh, iou]

    # Add normalized box coordinates if image size provided
    if image_size is not None:
        h, w = image_size
        scale = torch.tensor([w, h, w, h], dtype=subj_boxes.dtype, device=subj_boxes.device)
        norm_subj = subj_boxes / scale
        norm_obj = obj_boxes / scale
        features.extend([
            norm_subj[:, 0], norm_subj[:, 1], norm_subj[:, 2], norm_subj[:, 3],
            norm_obj[:, 0], norm_obj[:, 1], norm_obj[:, 2], norm_obj[:, 3]
        ])

    return torch.stack(features, dim=1)


def _compute_spatial_features_numpy(
    subj_boxes: np.ndarray,
    obj_boxes: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """NumPy implementation of spatial features."""
    # Box dimensions
    subj_w = subj_boxes[:, 2] - subj_boxes[:, 0]
    subj_h = subj_boxes[:, 3] - subj_boxes[:, 1]
    obj_w = obj_boxes[:, 2] - obj_boxes[:, 0]
    obj_h = obj_boxes[:, 3] - obj_boxes[:, 1]

    # Avoid division by zero
    subj_w = np.maximum(subj_w, 1e-8)
    subj_h = np.maximum(subj_h, 1e-8)
    obj_w = np.maximum(obj_w, 1e-8)
    obj_h = np.maximum(obj_h, 1e-8)

    # Centers
    subj_cx = (subj_boxes[:, 0] + subj_boxes[:, 2]) / 2
    subj_cy = (subj_boxes[:, 1] + subj_boxes[:, 3]) / 2
    obj_cx = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2
    obj_cy = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2

    # Relative displacement
    dx = (obj_cx - subj_cx) / subj_w
    dy = (obj_cy - subj_cy) / subj_h

    # Relative size
    dw = np.log(obj_w / subj_w)
    dh = np.log(obj_h / subj_h)

    # IoU
    iou = box_iou_pairwise(subj_boxes, obj_boxes)

    features = [dx, dy, dw, dh, iou]

    if image_size is not None:
        h, w = image_size
        scale = np.array([w, h, w, h])
        norm_subj = subj_boxes / scale
        norm_obj = obj_boxes / scale
        features.extend([
            norm_subj[:, 0], norm_subj[:, 1], norm_subj[:, 2], norm_subj[:, 3],
            norm_obj[:, 0], norm_obj[:, 1], norm_obj[:, 2], norm_obj[:, 3]
        ])

    return np.stack(features, axis=1)


def clip_boxes_to_image(
    boxes: ArrayLike,
    image_size: Tuple[int, int]
) -> ArrayLike:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: [N, 4] boxes
        image_size: (height, width)

    Returns:
        [N, 4] clipped boxes
    """
    h, w = image_size

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)
    else:
        boxes = boxes.copy()
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    return boxes


def box_center(boxes: ArrayLike) -> ArrayLike:
    """
    Compute center of boxes.

    Args:
        boxes: [..., 4] boxes

    Returns:
        [..., 2] centers (cx, cy)
    """
    if isinstance(boxes, torch.Tensor):
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        return torch.stack([cx, cy], dim=-1)
    else:
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        return np.stack([cx, cy], axis=-1)


def box_wh(boxes: ArrayLike) -> ArrayLike:
    """
    Compute width and height of boxes.

    Args:
        boxes: [..., 4] boxes

    Returns:
        [..., 2] (width, height)
    """
    if isinstance(boxes, torch.Tensor):
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        return torch.stack([w, h], dim=-1)
    else:
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        return np.stack([w, h], axis=-1)
