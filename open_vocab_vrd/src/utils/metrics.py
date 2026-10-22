"""Recall@K and mean-Recall@K for VRD / scene graph evaluation.

Evaluation mode: *triplet-level*.
Each ground-truth sample is a (subject, predicate, object) triplet.
A prediction "hits" a GT triplet if subject, predicate, and object all match
and the prediction appears in the top-K ranked predictions for that image.

This is an approximation of the standard SGDet/PredCls recall used in
Neural Motifs and related papers; it is noted as approximate in the output.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np


def _hit(pred_triplet: tuple, gt_set: set) -> bool:
    return pred_triplet in gt_set


def compute_recall_at_k(
    predictions: List[Dict],
    ground_truths: List[Dict],
    k: int,
) -> float:
    """Compute Recall@K across all images.

    Each prediction dict must have keys: image_id, subject, predicate, object, confidence.
    Each ground_truth dict must have keys: image_id, subject, predicate, object.

    Returns fraction of GT triplets recalled in the top-K predictions per image.
    """
    # Group GT by image
    gt_by_image: Dict[int, set] = defaultdict(set)
    for gt in ground_truths:
        key = (gt["subject"].lower(), gt["predicate"].lower(), gt["object"].lower())
        gt_by_image[gt["image_id"]].add(key)

    # Group predictions by image, sort by confidence desc
    pred_by_image: Dict[int, List] = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)
    for img_id in pred_by_image:
        pred_by_image[img_id].sort(key=lambda x: x["confidence"], reverse=True)

    total_gt = 0
    total_hit = 0

    for img_id, gt_set in gt_by_image.items():
        top_k_preds = pred_by_image.get(img_id, [])[:k]
        hit_set = set()
        for p in top_k_preds:
            key = (p["subject"].lower(), p["predicate"].lower(), p["object"].lower())
            if key in gt_set:
                hit_set.add(key)
        total_gt += len(gt_set)
        total_hit += len(hit_set)

    if total_gt == 0:
        return 0.0
    return total_hit / total_gt


def compute_mean_recall_at_k(
    predictions: List[Dict],
    ground_truths: List[Dict],
    k: int,
    pred_vocab: List[str],
) -> float:
    """Mean Recall@K — per-predicate recall averaged across predicate classes.

    Predicates with zero GT instances are excluded from the mean.
    """
    # Per-predicate GT sets keyed by image_id
    gt_by_pred_image: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for gt in ground_truths:
        pred_key = gt["predicate"].lower()
        triplet = (gt["subject"].lower(), gt["predicate"].lower(), gt["object"].lower())
        gt_by_pred_image[pred_key][gt["image_id"]].add(triplet)

    pred_by_image: Dict[int, List] = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)
    for img_id in pred_by_image:
        pred_by_image[img_id].sort(key=lambda x: x["confidence"], reverse=True)

    per_pred_recalls = []
    for pred_class in pred_vocab:
        pred_class_lower = pred_class.lower()
        if pred_class_lower not in gt_by_pred_image:
            continue
        total_gt = 0
        total_hit = 0
        for img_id, gt_set in gt_by_pred_image[pred_class_lower].items():
            top_k = pred_by_image.get(img_id, [])[:k]
            hit_set = set()
            for p in top_k:
                key = (p["subject"].lower(), p["predicate"].lower(), p["object"].lower())
                if key in gt_set:
                    hit_set.add(key)
            total_gt += len(gt_set)
            total_hit += len(hit_set)
        if total_gt > 0:
            per_pred_recalls.append(total_hit / total_gt)

    if not per_pred_recalls:
        return 0.0
    return float(np.mean(per_pred_recalls))


def compute_all_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    pred_vocab: List[str],
    recall_ks: List[int] = (20, 50, 100),
) -> Dict[str, float]:
    """Compute R@K and mR@K for all K values. Returns a flat metrics dict."""
    metrics = {}
    for k in recall_ks:
        metrics[f"R@{k}"] = compute_recall_at_k(predictions, ground_truths, k)
        metrics[f"mR@{k}"] = compute_mean_recall_at_k(predictions, ground_truths, k, pred_vocab)
    return metrics


def per_predicate_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    pred_vocab: List[str],
    k: int = 50,
) -> Dict[str, float]:
    """Return recall@k broken down by predicate class."""
    gt_by_pred_image: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for gt in ground_truths:
        key = gt["predicate"].lower()
        triplet = (gt["subject"].lower(), gt["predicate"].lower(), gt["object"].lower())
        gt_by_pred_image[key][gt["image_id"]].add(triplet)

    pred_by_image: Dict[int, List] = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)
    for img_id in pred_by_image:
        pred_by_image[img_id].sort(key=lambda x: x["confidence"], reverse=True)

    result = {}
    for pred_class in pred_vocab:
        pc = pred_class.lower()
        if pc not in gt_by_pred_image:
            result[pred_class] = None  # no GT examples
            continue
        total_gt, total_hit = 0, 0
        for img_id, gt_set in gt_by_pred_image[pc].items():
            top_k = pred_by_image.get(img_id, [])[:k]
            hit_set = {
                (p["subject"].lower(), p["predicate"].lower(), p["object"].lower())
                for p in top_k
                if (p["subject"].lower(), p["predicate"].lower(), p["object"].lower()) in gt_set
            }
            total_gt += len(gt_set)
            total_hit += len(hit_set)
        result[pred_class] = total_hit / total_gt if total_gt > 0 else 0.0
    return result
