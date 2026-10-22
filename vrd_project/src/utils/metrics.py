"""
VRD Evaluation Metrics
======================

Implements Recall@K and mean Recall@K (mR@K) metrics for VRD evaluation.
These are the standard metrics used in Visual Relationship Detection literature.

References:
- Visual Relationship Detection with Language Priors (Lu et al., 2016)
- Neural Motifs (Zellers et al., 2018)
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import torch


def box_union_single(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute the union (enclosing) box of two boxes.

    Args:
        box1, box2: Boxes as [x1, y1, x2, y2]

    Returns:
        Union box as [x1, y1, x2, y2]
    """
    return np.array([
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    ])


def triplet_match(
    pred_triplet: Tuple[int, int, int],
    gt_triplet: Tuple[int, int, int],
    pred_boxes: Tuple[np.ndarray, np.ndarray],
    gt_boxes: Tuple[np.ndarray, np.ndarray],
    iou_threshold: float = 0.5,
    mode: str = 'predcls'
) -> bool:
    """
    Check if a predicted triplet matches a ground truth triplet.

    Args:
        pred_triplet: (subj_label, pred_label, obj_label)
        gt_triplet: (subj_label, pred_label, obj_label)
        pred_boxes: (subj_box, obj_box) as numpy arrays [x1, y1, x2, y2]
        gt_boxes: (subj_box, obj_box) as numpy arrays
        iou_threshold: IoU threshold for box matching
        mode: Evaluation mode ('predcls', 'sgcls', 'sgdet', 'phrase_det', 'rel_det')

    Returns:
        True if triplet matches under the given mode
    """
    subj_pred, pred_pred, obj_pred = pred_triplet
    subj_gt, pred_gt, obj_gt = gt_triplet
    pred_subj_box, pred_obj_box = pred_boxes
    gt_subj_box, gt_obj_box = gt_boxes

    if mode == 'predcls':
        # Only check predicate match (boxes and labels are given)
        return pred_pred == pred_gt

    elif mode == 'sgcls':
        # Check labels and predicate match (boxes are given)
        return (subj_pred == subj_gt and
                obj_pred == obj_gt and
                pred_pred == pred_gt)

    elif mode in ('sgdet', 'rel_det'):
        # Check everything: boxes must have IoU >= threshold, labels and predicate must match
        # rel_det is the standard Relationship Detection task (identical to sgdet matching)
        subj_iou = box_iou_single(pred_subj_box, gt_subj_box)
        obj_iou = box_iou_single(pred_obj_box, gt_obj_box)

        return (subj_iou >= iou_threshold and
                obj_iou >= iou_threshold and
                subj_pred == subj_gt and
                obj_pred == obj_gt and
                pred_pred == pred_gt)

    elif mode == 'phrase_det':
        # Phrase Detection: match using union box IoU + all labels must match
        pred_union = box_union_single(pred_subj_box, pred_obj_box)
        gt_union = box_union_single(gt_subj_box, gt_obj_box)
        union_iou = box_iou_single(pred_union, gt_union)

        return (union_iou >= iou_threshold and
                subj_pred == subj_gt and
                obj_pred == obj_gt and
                pred_pred == pred_gt)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def box_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Args:
        box1, box2: Boxes as [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_recall_and_mean_recall_at_k(
    predictions: List[Dict],
    ground_truths: List[Dict],
    k_values: List[int] = [20, 50, 100],
    num_predicates: int = 50,
    mode: str = 'predcls',
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute both Recall@K and mean Recall@K in a single pass.

    Replaces calling compute_recall_at_k() and compute_mean_recall_at_k()
    separately, which would run the same triplet_match loop twice.
    Boxes are pre-converted to numpy arrays once per image rather than
    inside the inner loop.

    Returns:
        Dictionary with R@K and mR@K for each K value.
    """
    recalls = {f'R@{k}': 0.0 for k in k_values}
    predicate_hits = {k: defaultdict(int) for k in k_values}
    predicate_counts: Dict[int, int] = defaultdict(int)
    total_gt_relations = 0

    for pred, gt in zip(predictions, ground_truths):
        if len(gt['triplets']) == 0:
            continue

        # Count GT predicates for mR@K denominator
        for triplet in gt['triplets']:
            predicate_counts[triplet[1]] += 1

        # Sort predictions by score once per image
        if len(pred['triplets']) > 0:
            sorted_indices = np.argsort(-np.array(pred['scores']))
        else:
            sorted_indices = np.empty(0, dtype=np.intp)

        # Pre-convert ALL boxes to numpy arrays once — not inside the triple loop
        n_pred = len(pred['triplets'])
        n_gt   = len(gt['triplets'])
        pred_subj_arr = np.array(pred['subj_boxes']) if n_pred > 0 else np.empty((0, 4))
        pred_obj_arr  = np.array(pred['obj_boxes'])  if n_pred > 0 else np.empty((0, 4))
        gt_subj_arr   = np.array(gt['subj_boxes'])   if n_gt   > 0 else np.empty((0, 4))
        gt_obj_arr    = np.array(gt['obj_boxes'])    if n_gt   > 0 else np.empty((0, 4))

        for k in k_values:
            gt_matched = [False] * n_gt
            hits = 0

            for i in sorted_indices[:k]:
                pred_triplet = pred['triplets'][i]
                p_boxes = (pred_subj_arr[i], pred_obj_arr[i])

                for j, gt_triplet in enumerate(gt['triplets']):
                    if gt_matched[j]:
                        continue

                    if triplet_match(
                        pred_triplet, gt_triplet,
                        p_boxes, (gt_subj_arr[j], gt_obj_arr[j]),
                        iou_threshold, mode
                    ):
                        gt_matched[j] = True
                        hits += 1
                        predicate_hits[k][gt_triplet[1]] += 1
                        break

            recalls[f'R@{k}'] += hits

        total_gt_relations += n_gt

    # Normalise R@K
    if total_gt_relations > 0:
        for k in k_values:
            recalls[f'R@{k}'] /= total_gt_relations

    # Compute mR@K with vectorised numpy division
    mean_recalls: Dict[str, float] = {}
    counts_arr = np.array([predicate_counts[p] for p in range(num_predicates)], dtype=np.float64)
    for k in k_values:
        hits_arr = np.array([predicate_hits[k][p] for p in range(num_predicates)], dtype=np.float64)
        valid = counts_arr > 0
        mean_recalls[f'mR@{k}'] = float(np.mean(hits_arr[valid] / counts_arr[valid])) if valid.any() else 0.0

    return {**recalls, **mean_recalls}


def compute_recall_at_k(
    predictions: List[Dict],
    ground_truths: List[Dict],
    k_values: List[int] = [20, 50, 100],
    mode: str = 'predcls',
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute Recall@K. Thin wrapper kept for backward compatibility."""
    combined = compute_recall_and_mean_recall_at_k(
        predictions, ground_truths, k_values, 1, mode, iou_threshold
    )
    return {k: v for k, v in combined.items() if k.startswith('R@')}


def compute_mean_recall_at_k(
    predictions: List[Dict],
    ground_truths: List[Dict],
    k_values: List[int] = [20, 50, 100],
    num_predicates: int = 50,
    mode: str = 'predcls',
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute mean Recall@K. Thin wrapper kept for backward compatibility."""
    combined = compute_recall_and_mean_recall_at_k(
        predictions, ground_truths, k_values, num_predicates, mode, iou_threshold
    )
    return {k: v for k, v in combined.items() if k.startswith('mR@')}


class VRDMetrics:
    """
    Comprehensive VRD metric computation class.

    Accumulates predictions and ground truths, then computes all metrics.
    """

    def __init__(
        self,
        k_values: List[int] = [20, 50, 100],
        num_predicates: int = 50,
        mode: str = 'predcls',
        iou_threshold: float = 0.5
    ):
        """
        Initialize VRD metrics.

        Args:
            k_values: K values for Recall@K
            num_predicates: Number of predicate classes
            mode: Evaluation mode
            iou_threshold: IoU threshold for SGDet
        """
        self.k_values = k_values
        self.num_predicates = num_predicates
        self.mode = mode
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions and ground truths."""
        self.predictions = []
        self.ground_truths = []

    def update(
        self,
        pred_triplets: List[Tuple[int, int, int]],
        pred_scores: List[float],
        pred_subj_boxes: List[np.ndarray],
        pred_obj_boxes: List[np.ndarray],
        gt_triplets: List[Tuple[int, int, int]],
        gt_subj_boxes: List[np.ndarray],
        gt_obj_boxes: List[np.ndarray]
    ) -> None:
        """
        Add predictions and ground truths for one image.

        Args:
            pred_triplets: List of predicted triplets
            pred_scores: Confidence scores for each prediction
            pred_subj_boxes: Subject boxes for predictions
            pred_obj_boxes: Object boxes for predictions
            gt_triplets: Ground truth triplets
            gt_subj_boxes: GT subject boxes
            gt_obj_boxes: GT object boxes
        """
        self.predictions.append({
            'triplets': pred_triplets,
            'scores': pred_scores,
            'subj_boxes': pred_subj_boxes,
            'obj_boxes': pred_obj_boxes
        })
        self.ground_truths.append({
            'triplets': gt_triplets,
            'subj_boxes': gt_subj_boxes,
            'obj_boxes': gt_obj_boxes
        })

    def update_batch(
        self,
        batch_predictions: List[Dict],
        batch_ground_truths: List[Dict]
    ) -> None:
        """
        Add predictions and ground truths for a batch of images.

        Args:
            batch_predictions: List of prediction dicts
            batch_ground_truths: List of GT dicts
        """
        self.predictions.extend(batch_predictions)
        self.ground_truths.extend(batch_ground_truths)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary with all metric values
        """
        metrics = {}

        # R@K and mR@K in a single pass (avoids running triplet_match twice)
        combined = compute_recall_and_mean_recall_at_k(
            self.predictions,
            self.ground_truths,
            self.k_values,
            self.num_predicates,
            self.mode,
            self.iou_threshold
        )
        metrics.update(combined)

        # Additional statistics
        metrics['num_images'] = len(self.predictions)
        metrics['total_gt_relations'] = sum(
            len(gt['triplets']) for gt in self.ground_truths
        )
        metrics['total_predictions'] = sum(
            len(pred['triplets']) for pred in self.predictions
        )

        # Head/Body/Tail breakdown
        hbt = self._compute_head_body_tail(k=50)
        metrics.update(hbt)

        return metrics

    def _compute_head_body_tail(self, k: int = 50) -> Dict[str, float]:
        """
        Compute R@K separately for Head, Body, and Tail predicates.

        Predicates are split by their frequency in ground truth:
          - Head: top ~33% most frequent predicates
          - Body: middle ~33%
          - Tail: bottom ~33% (rarest)

        This is the critical metric for proving long-tail improvement.
        """
        # 1. Count predicate frequencies across all GT
        pred_freq = defaultdict(int)
        for gt in self.ground_truths:
            for triplet in gt['triplets']:
                pred_freq[triplet[1]] += 1

        if not pred_freq:
            return {}

        # 2. Sort predicates by frequency and split into 3 groups
        sorted_preds = sorted(pred_freq.keys(), key=lambda p: pred_freq[p], reverse=True)
        n = len(sorted_preds)
        split1 = max(n // 3, 1)
        split2 = max(2 * n // 3, split1 + 1)

        head_preds = set(sorted_preds[:split1])
        body_preds = set(sorted_preds[split1:split2])
        tail_preds = set(sorted_preds[split2:])

        # 3. Compute per-group recall
        group_hits = {'head': 0, 'body': 0, 'tail': 0}
        group_total = {'head': 0, 'body': 0, 'tail': 0}

        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(gt['triplets']) == 0:
                continue

            # Sort predictions by score
            if len(pred['triplets']) > 0:
                sorted_indices = np.argsort(-np.array(pred['scores']))[:k]
            else:
                sorted_indices = np.empty(0, dtype=np.intp)

            n_gt = len(gt['triplets'])
            pred_subj_arr = np.array(pred['subj_boxes']) if len(pred['triplets']) > 0 else np.empty((0, 4))
            pred_obj_arr = np.array(pred['obj_boxes']) if len(pred['triplets']) > 0 else np.empty((0, 4))
            gt_subj_arr = np.array(gt['subj_boxes']) if n_gt > 0 else np.empty((0, 4))
            gt_obj_arr = np.array(gt['obj_boxes']) if n_gt > 0 else np.empty((0, 4))

            gt_matched = [False] * n_gt

            for i in sorted_indices:
                pred_triplet = pred['triplets'][i]
                p_boxes = (pred_subj_arr[i], pred_obj_arr[i])

                for j, gt_triplet in enumerate(gt['triplets']):
                    if gt_matched[j]:
                        continue
                    if triplet_match(
                        pred_triplet, gt_triplet,
                        p_boxes, (gt_subj_arr[j], gt_obj_arr[j]),
                        self.iou_threshold, self.mode
                    ):
                        gt_matched[j] = True
                        pred_id = gt_triplet[1]
                        if pred_id in head_preds:
                            group_hits['head'] += 1
                        elif pred_id in body_preds:
                            group_hits['body'] += 1
                        elif pred_id in tail_preds:
                            group_hits['tail'] += 1
                        break

            # Count GT per group
            for gt_triplet in gt['triplets']:
                pred_id = gt_triplet[1]
                if pred_id in head_preds:
                    group_total['head'] += 1
                elif pred_id in body_preds:
                    group_total['body'] += 1
                elif pred_id in tail_preds:
                    group_total['tail'] += 1

        results = {}
        for group in ['head', 'body', 'tail']:
            if group_total[group] > 0:
                results[f'R@{k}_{group}'] = group_hits[group] / group_total[group]
            else:
                results[f'R@{k}_{group}'] = 0.0

        results['head_predicates'] = len(head_preds)
        results['body_predicates'] = len(body_preds)
        results['tail_predicates'] = len(tail_preds)

        return results

    def __repr__(self) -> str:
        metrics = self.compute()
        lines = [f"VRDMetrics (mode={self.mode}):"]
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return '\n'.join(lines)


def format_metrics(metrics: Dict[str, float], mode: str = 'predcls') -> str:
    """
    Format metrics for display.

    Args:
        metrics: Dictionary of metric values
        mode: Evaluation mode

    Returns:
        Formatted string
    """
    lines = [f"\n{'='*50}", f"Evaluation Results ({mode.upper()})", '='*50]

    # Recall metrics
    for k in [20, 50, 100]:
        if f'R@{k}' in metrics:
            lines.append(f"R@{k}:  {metrics[f'R@{k}']*100:.2f}%")

    lines.append('-'*50)

    # Mean Recall metrics
    for k in [20, 50, 100]:
        if f'mR@{k}' in metrics:
            lines.append(f"mR@{k}: {metrics[f'mR@{k}']*100:.2f}%")

    # Head/Body/Tail breakdown
    if 'R@50_head' in metrics:
        lines.append('-'*50)
        lines.append('Head/Body/Tail Breakdown (R@50):')
        lines.append(f"  Head ({int(metrics.get('head_predicates', 0))} preds): {metrics['R@50_head']*100:.2f}%")
        lines.append(f"  Body ({int(metrics.get('body_predicates', 0))} preds): {metrics['R@50_body']*100:.2f}%")
        lines.append(f"  Tail ({int(metrics.get('tail_predicates', 0))} preds): {metrics['R@50_tail']*100:.2f}%")

    lines.append('-'*50)

    # Statistics
    if 'num_images' in metrics:
        lines.append(f"Images evaluated: {metrics['num_images']}")
    if 'total_gt_relations' in metrics:
        lines.append(f"Total GT relations: {metrics['total_gt_relations']}")
    if 'total_predictions' in metrics:
        lines.append(f"Total predictions: {metrics['total_predictions']}")

    lines.append('='*50 + '\n')

    return '\n'.join(lines)
