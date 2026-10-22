"""
Tests for VRD Metrics
======================

Tests for Recall@K and mean Recall@K computation.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import (
    compute_recall_at_k, compute_mean_recall_at_k,
    VRDMetrics, triplet_match, box_iou_single
)


class TestBoxIoU:
    """Tests for box IoU computation."""

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes."""
        box = np.array([10, 10, 50, 50])
        iou = box_iou_single(box, box)
        assert abs(iou - 1.0) < 1e-6

    def test_iou_non_overlapping(self):
        """Test IoU of non-overlapping boxes."""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        iou = box_iou_single(box1, box2)
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = np.array([0, 0, 20, 20])
        box2 = np.array([10, 10, 30, 30])

        # Intersection: 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        # IoU: 100/700 = 0.1428...

        iou = box_iou_single(box1, box2)
        expected = 100 / 700
        assert abs(iou - expected) < 1e-5


class TestTripletMatch:
    """Tests for triplet matching."""

    def test_predcls_mode_match(self):
        """Test triplet matching in PredCls mode."""
        # In PredCls, only predicate needs to match
        pred_triplet = (1, 5, 2)  # (subj, pred, obj)
        gt_triplet = (1, 5, 2)

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))

        assert triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='predcls'
        )

    def test_predcls_mode_no_match(self):
        """Test triplet not matching in PredCls mode."""
        pred_triplet = (1, 5, 2)
        gt_triplet = (1, 6, 2)  # Different predicate

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))

        assert not triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='predcls'
        )

    def test_sgcls_mode_match(self):
        """Test triplet matching in SGCls mode."""
        # In SGCls, labels and predicate need to match
        pred_triplet = (1, 5, 2)
        gt_triplet = (1, 5, 2)

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))

        assert triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='sgcls'
        )

    def test_sgcls_mode_no_match(self):
        """Test triplet not matching in SGCls mode."""
        pred_triplet = (1, 5, 2)
        gt_triplet = (3, 5, 2)  # Different subject label

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))

        assert not triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='sgcls'
        )

    def test_sgdet_mode_match(self):
        """Test triplet matching in SGDet mode."""
        # In SGDet, boxes must have IoU >= threshold, and labels/pred must match
        pred_triplet = (1, 5, 2)
        gt_triplet = (1, 5, 2)

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))

        assert triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='sgdet',
            iou_threshold=0.5
        )

    def test_sgdet_mode_no_match_iou(self):
        """Test triplet not matching in SGDet due to low IoU."""
        pred_triplet = (1, 5, 2)
        gt_triplet = (1, 5, 2)

        pred_boxes = (np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        gt_boxes = (np.array([100, 100, 110, 110]), np.array([120, 120, 130, 130]))

        assert not triplet_match(
            pred_triplet, gt_triplet,
            pred_boxes, gt_boxes,
            mode='sgdet',
            iou_threshold=0.5
        )


class TestRecallAtK:
    """Tests for Recall@K computation."""

    def test_perfect_predictions(self):
        """Test Recall@K with perfect predictions."""
        predictions = [{
            'triplets': [(1, 5, 2)],
            'scores': [0.9],
            'subj_boxes': [np.array([0, 0, 10, 10])],
            'obj_boxes': [np.array([20, 20, 30, 30])]
        }]

        ground_truths = [{
            'triplets': [(1, 5, 2)],
            'subj_boxes': [np.array([0, 0, 10, 10])],
            'obj_boxes': [np.array([20, 20, 30, 30])]
        }]

        recalls = compute_recall_at_k(
            predictions, ground_truths,
            k_values=[20, 50, 100],
            mode='predcls'
        )

        assert recalls['R@20'] == 1.0
        assert recalls['R@50'] == 1.0
        assert recalls['R@100'] == 1.0

    def test_no_predictions(self):
        """Test Recall@K with no predictions."""
        predictions = [{
            'triplets': [],
            'scores': [],
            'subj_boxes': [],
            'obj_boxes': []
        }]

        ground_truths = [{
            'triplets': [(1, 5, 2)],
            'subj_boxes': [np.array([0, 0, 10, 10])],
            'obj_boxes': [np.array([20, 20, 30, 30])]
        }]

        recalls = compute_recall_at_k(
            predictions, ground_truths,
            k_values=[20, 50, 100],
            mode='predcls'
        )

        assert recalls['R@20'] == 0.0
        assert recalls['R@50'] == 0.0
        assert recalls['R@100'] == 0.0

    def test_partial_recall(self):
        """Test Recall@K with partial matches."""
        # 2 GT relations, only 1 correctly predicted
        predictions = [{
            'triplets': [(1, 5, 2), (1, 6, 3)],  # Only first matches
            'scores': [0.9, 0.8],
            'subj_boxes': [np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])],
            'obj_boxes': [np.array([20, 20, 30, 30]), np.array([40, 40, 50, 50])]
        }]

        ground_truths = [{
            'triplets': [(1, 5, 2), (1, 7, 3)],  # Second has different predicate
            'subj_boxes': [np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])],
            'obj_boxes': [np.array([20, 20, 30, 30]), np.array([40, 40, 50, 50])]
        }]

        recalls = compute_recall_at_k(
            predictions, ground_truths,
            k_values=[20],
            mode='predcls'
        )

        assert recalls['R@20'] == 0.5  # 1/2

    def test_multiple_images(self):
        """Test Recall@K across multiple images."""
        predictions = [
            {
                'triplets': [(1, 5, 2)],
                'scores': [0.9],
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            },
            {
                'triplets': [(3, 8, 4)],
                'scores': [0.8],
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            }
        ]

        ground_truths = [
            {
                'triplets': [(1, 5, 2)],
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            },
            {
                'triplets': [(3, 9, 4)],  # Different predicate - no match
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            }
        ]

        recalls = compute_recall_at_k(
            predictions, ground_truths,
            k_values=[20],
            mode='predcls'
        )

        # 1 match out of 2 GT relations
        assert recalls['R@20'] == 0.5


class TestMeanRecallAtK:
    """Tests for mean Recall@K computation."""

    def test_mean_recall_uniform(self):
        """Test mR@K when all predicates have same recall."""
        # Create predictions where each predicate is correctly predicted
        predictions = []
        ground_truths = []

        for pred_id in range(1, 6):  # 5 predicates
            predictions.append({
                'triplets': [(1, pred_id, 2)],
                'scores': [0.9],
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            })
            ground_truths.append({
                'triplets': [(1, pred_id, 2)],
                'subj_boxes': [np.array([0, 0, 10, 10])],
                'obj_boxes': [np.array([20, 20, 30, 30])]
            })

        mrecalls = compute_mean_recall_at_k(
            predictions, ground_truths,
            k_values=[20],
            num_predicates=6,  # Including 0
            mode='predcls'
        )

        # All predicates have recall 1.0
        assert mrecalls['mR@20'] == 1.0


class TestVRDMetrics:
    """Tests for VRDMetrics class."""

    def test_metrics_class_creation(self):
        """Test VRDMetrics creation."""
        metrics = VRDMetrics(
            k_values=[20, 50, 100],
            num_predicates=51,
            mode='predcls'
        )
        assert metrics is not None

    def test_metrics_update_and_compute(self):
        """Test metrics update and compute."""
        metrics = VRDMetrics(
            k_values=[20],
            num_predicates=10,
            mode='predcls'
        )

        # Add predictions
        metrics.update(
            pred_triplets=[(1, 5, 2)],
            pred_scores=[0.9],
            pred_subj_boxes=[np.array([0, 0, 10, 10])],
            pred_obj_boxes=[np.array([20, 20, 30, 30])],
            gt_triplets=[(1, 5, 2)],
            gt_subj_boxes=[np.array([0, 0, 10, 10])],
            gt_obj_boxes=[np.array([20, 20, 30, 30])]
        )

        result = metrics.compute()

        assert 'R@20' in result
        assert 'mR@20' in result
        assert result['R@20'] == 1.0

    def test_metrics_reset(self):
        """Test metrics reset."""
        metrics = VRDMetrics(k_values=[20], num_predicates=10)

        metrics.update(
            pred_triplets=[(1, 5, 2)],
            pred_scores=[0.9],
            pred_subj_boxes=[np.array([0, 0, 10, 10])],
            pred_obj_boxes=[np.array([20, 20, 30, 30])],
            gt_triplets=[(1, 5, 2)],
            gt_subj_boxes=[np.array([0, 0, 10, 10])],
            gt_obj_boxes=[np.array([20, 20, 30, 30])]
        )

        metrics.reset()

        result = metrics.compute()
        assert result['num_images'] == 0
        assert result['total_gt_relations'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
