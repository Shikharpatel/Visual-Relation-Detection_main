"""13-dimensional spatial feature encoding for subject-object pairs.

Feature vector layout (matches config spatial.feature_dim = 13):
  [0:4]  subject box normalized: [x1/W, y1/H, x2/W, y2/H]
  [4:8]  object  box normalized: [x1/W, y1/H, x2/W, y2/H]
  [8]    relative center dx: (cx_obj - cx_subj) / W
  [9]    relative center dy: (cy_obj - cy_subj) / H
  [10]   log width  ratio:  log(w_obj / w_subj)   (clipped to [-5, 5])
  [11]   log height ratio:  log(h_obj / h_subj)   (clipped to [-5, 5])
  [12]   IoU between the two boxes
"""

from typing import List, Union

import numpy as np


def _box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU for two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_spatial_features(
    subj_box: Union[List[float], np.ndarray],
    obj_box: Union[List[float], np.ndarray],
    image_w: float,
    image_h: float,
) -> np.ndarray:
    """Return a 13-D spatial feature vector for one (subject, object) pair.

    Args:
        subj_box: [x1, y1, x2, y2] in pixel coords.
        obj_box:  [x1, y1, x2, y2] in pixel coords.
        image_w:  Image width in pixels.
        image_h:  Image height in pixels.

    Returns:
        np.ndarray of shape (13,) dtype float32.
    """
    subj = np.asarray(subj_box, dtype=np.float32)
    obj  = np.asarray(obj_box,  dtype=np.float32)

    # Avoid division by zero on degenerate images
    W = max(image_w, 1.0)
    H = max(image_h, 1.0)

    # Normalized boxes
    s_norm = np.array([subj[0] / W, subj[1] / H, subj[2] / W, subj[3] / H], dtype=np.float32)
    o_norm = np.array([obj[0]  / W, obj[1]  / H, obj[2]  / W, obj[3]  / H], dtype=np.float32)

    # Centers
    cx_s = (subj[0] + subj[2]) / 2.0
    cy_s = (subj[1] + subj[3]) / 2.0
    cx_o = (obj[0]  + obj[2])  / 2.0
    cy_o = (obj[1]  + obj[3])  / 2.0

    dx = (cx_o - cx_s) / W
    dy = (cy_o - cy_s) / H

    # Size ratios (log-scale, clipped to avoid extreme values)
    w_s = max(subj[2] - subj[0], 1.0)
    h_s = max(subj[3] - subj[1], 1.0)
    w_o = max(obj[2]  - obj[0],  1.0)
    h_o = max(obj[3]  - obj[1],  1.0)

    log_w_ratio = float(np.clip(np.log(w_o / w_s), -5.0, 5.0))
    log_h_ratio = float(np.clip(np.log(h_o / h_s), -5.0, 5.0))

    iou = _box_iou(subj, obj)

    feat = np.concatenate([
        s_norm,
        o_norm,
        [dx, dy, log_w_ratio, log_h_ratio, iou],
    ]).astype(np.float32)

    assert feat.shape == (13,), f"Expected (13,), got {feat.shape}"
    return feat


def compute_spatial_features_batch(
    subj_boxes: np.ndarray,
    obj_boxes: np.ndarray,
    image_ws: np.ndarray,
    image_hs: np.ndarray,
) -> np.ndarray:
    """Vectorised version for a batch of pairs.

    Args:
        subj_boxes: (N, 4) float array [x1,y1,x2,y2].
        obj_boxes:  (N, 4) float array [x1,y1,x2,y2].
        image_ws:   (N,) float array of image widths.
        image_hs:   (N,) float array of image heights.

    Returns:
        (N, 13) float32 array.
    """
    N = len(subj_boxes)
    out = np.zeros((N, 13), dtype=np.float32)
    for i in range(N):
        out[i] = compute_spatial_features(
            subj_boxes[i], obj_boxes[i], image_ws[i], image_hs[i]
        )
    return out
