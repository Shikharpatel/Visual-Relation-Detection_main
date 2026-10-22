"""
Tests for VRD Models
=====================

Tests for model forward passes and components.
"""

import sys
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scene_graph_model import SceneGraphModel, SimpleVRDModel, create_model
from src.models.relation_head import RelationHead, SpatialEncoder
from src.models.frequency_bias import FrequencyBias


class TestRelationHead:
    """Tests for RelationHead module."""

    def test_relation_head_creation(self):
        """Test that relation head can be created."""
        head = RelationHead(
            visual_dim=256,
            spatial_dim=128,
            hidden_dim=512,
            num_predicates=51
        )
        assert head is not None

    def test_relation_head_forward(self):
        """Test relation head forward pass."""
        head = RelationHead(
            visual_dim=256,
            spatial_dim=128,
            hidden_dim=512,
            num_predicates=51
        )

        batch_size = 10
        subj_features = torch.randn(batch_size, 256)
        obj_features = torch.randn(batch_size, 256)
        union_features = torch.randn(batch_size, 256)
        subj_boxes = torch.tensor([[10, 10, 50, 50]] * batch_size).float()
        obj_boxes = torch.tensor([[60, 60, 100, 100]] * batch_size).float()

        logits = head(
            subj_features, obj_features,
            subj_boxes, obj_boxes,
            union_features=union_features
        )

        assert logits.shape == (batch_size, 51)

    def test_relation_head_no_union(self):
        """Test relation head without union features."""
        head = RelationHead(
            visual_dim=256,
            spatial_dim=128,
            hidden_dim=512,
            num_predicates=51,
            use_union_features=False
        )

        batch_size = 10
        subj_features = torch.randn(batch_size, 256)
        obj_features = torch.randn(batch_size, 256)
        subj_boxes = torch.tensor([[10, 10, 50, 50]] * batch_size).float()
        obj_boxes = torch.tensor([[60, 60, 100, 100]] * batch_size).float()

        logits = head(
            subj_features, obj_features,
            subj_boxes, obj_boxes
        )

        assert logits.shape == (batch_size, 51)

    def test_relation_head_no_spatial(self):
        """Test relation head without spatial features."""
        head = RelationHead(
            visual_dim=256,
            spatial_dim=128,
            hidden_dim=512,
            num_predicates=51,
            use_spatial=False
        )

        batch_size = 10
        subj_features = torch.randn(batch_size, 256)
        obj_features = torch.randn(batch_size, 256)
        union_features = torch.randn(batch_size, 256)
        subj_boxes = torch.tensor([[10, 10, 50, 50]] * batch_size).float()
        obj_boxes = torch.tensor([[60, 60, 100, 100]] * batch_size).float()

        logits = head(
            subj_features, obj_features,
            subj_boxes, obj_boxes,
            union_features=union_features
        )

        assert logits.shape == (batch_size, 51)


class TestSpatialEncoder:
    """Tests for SpatialEncoder module."""

    def test_spatial_encoder_creation(self):
        """Test spatial encoder creation."""
        encoder = SpatialEncoder(output_dim=128)
        assert encoder is not None

    def test_spatial_encoder_forward(self):
        """Test spatial encoder forward pass."""
        encoder = SpatialEncoder(output_dim=128, use_coords=True)

        batch_size = 10
        subj_boxes = torch.tensor([[10, 10, 50, 50]] * batch_size).float()
        obj_boxes = torch.tensor([[60, 60, 100, 100]] * batch_size).float()

        features = encoder(subj_boxes, obj_boxes, image_size=(480, 640))

        assert features.shape == (batch_size, 128)

    def test_spatial_encoder_without_coords(self):
        """Test spatial encoder without coordinate features."""
        encoder = SpatialEncoder(output_dim=64, use_coords=False)

        batch_size = 5
        subj_boxes = torch.randn(batch_size, 4).abs() * 100
        obj_boxes = torch.randn(batch_size, 4).abs() * 100

        # Make boxes valid
        subj_boxes[:, 2:] = subj_boxes[:, :2] + 50
        obj_boxes[:, 2:] = obj_boxes[:, :2] + 50

        features = encoder(subj_boxes, obj_boxes)

        assert features.shape == (batch_size, 64)


class TestFrequencyBias:
    """Tests for FrequencyBias module."""

    def test_frequency_bias_creation(self):
        """Test frequency bias creation."""
        bias = FrequencyBias(
            num_obj_classes=151,
            num_pred_classes=51,
            alpha=1.0
        )
        assert bias is not None
        assert not bias.is_fitted

    def test_frequency_bias_forward_unfitted(self):
        """Test frequency bias forward when not fitted."""
        bias = FrequencyBias(
            num_obj_classes=151,
            num_pred_classes=51
        )

        subj_labels = torch.tensor([1, 2, 3])
        obj_labels = torch.tensor([4, 5, 6])

        logits = bias(subj_labels, obj_labels)

        # Should return zeros when not fitted
        assert logits.shape == (3, 51)
        assert (logits == 0).all()

    def test_frequency_bias_fit(self):
        """Test frequency bias fitting."""
        bias = FrequencyBias(
            num_obj_classes=10,
            num_pred_classes=5
        )

        # Create dummy training data
        train_data = [
            {
                'objects': [
                    {'class_idx': 1},
                    {'class_idx': 2}
                ],
                'relations': [
                    {'subject_idx': 0, 'object_idx': 1, 'predicate_idx': 1}
                ]
            },
            {
                'objects': [
                    {'class_idx': 1},
                    {'class_idx': 2}
                ],
                'relations': [
                    {'subject_idx': 0, 'object_idx': 1, 'predicate_idx': 1},
                    {'subject_idx': 0, 'object_idx': 1, 'predicate_idx': 2}
                ]
            }
        ]

        bias.fit(train_data)

        assert bias.is_fitted

    def test_frequency_bias_forward_fitted(self):
        """Test frequency bias forward when fitted."""
        bias = FrequencyBias(
            num_obj_classes=10,
            num_pred_classes=5
        )

        # Manually set fitted state
        bias.freq_matrix.fill_(0.2)  # Uniform distribution
        bias.is_fitted.fill_(True)

        subj_labels = torch.tensor([1, 2, 3])
        obj_labels = torch.tensor([4, 5, 6])

        logits = bias(subj_labels, obj_labels)

        assert logits.shape == (3, 5)
        assert not (logits == 0).all()


class TestSceneGraphModel:
    """Tests for SceneGraphModel."""

    def test_model_creation(self):
        """Test model creation."""
        model = SceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls'
        )
        assert model is not None

    def test_model_forward_single(self):
        """Test model forward pass with single image."""
        model = SceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls',
            feature_dim=256,
            use_bias=False  # Disable bias for testing
        )

        num_objects = 5
        num_relations = 10

        obj_features = torch.randn(num_objects, 1024)
        boxes = torch.tensor([
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [110, 110, 150, 150],
            [160, 160, 200, 200],
            [210, 210, 250, 250]
        ]).float()
        labels = torch.tensor([1, 2, 3, 4, 5])
        relation_pairs = torch.tensor([
            [0, 1], [0, 2], [0, 3], [1, 0], [1, 2],
            [2, 0], [2, 3], [3, 1], [3, 4], [4, 0]
        ])

        outputs = model(
            obj_features, boxes, labels, relation_pairs
        )

        assert 'pred_logits' in outputs
        assert 'pred_scores' in outputs
        assert outputs['pred_logits'].shape == (num_relations, 51)
        assert outputs['pred_scores'].shape == (num_relations, 51)

    def test_model_forward_batch(self):
        """Test model forward pass with batch."""
        model = SceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls',
            use_bias=False
        )

        batch_size = 2
        max_objects = 5
        max_relations = 8

        obj_features = torch.randn(batch_size, max_objects, 1024)
        boxes = torch.randn(batch_size, max_objects, 4).abs() * 100
        boxes[:, :, 2:] = boxes[:, :, :2] + 50
        labels = torch.randint(1, 151, (batch_size, max_objects))
        relation_pairs = torch.randint(0, max_objects, (batch_size, max_relations, 2))
        object_mask = torch.ones(batch_size, max_objects, dtype=torch.bool)
        relation_mask = torch.ones(batch_size, max_relations, dtype=torch.bool)

        outputs = model(
            obj_features, boxes, labels, relation_pairs,
            object_mask, relation_mask
        )

        assert outputs['pred_logits'].shape == (batch_size, max_relations, 51)

    def test_model_loss_computation(self):
        """Test model loss computation."""
        model = SceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls',
            use_bias=False
        )

        batch_size = 2
        max_objects = 5
        max_relations = 8

        obj_features = torch.randn(batch_size, max_objects, 1024)
        boxes = torch.randn(batch_size, max_objects, 4).abs() * 100
        boxes[:, :, 2:] = boxes[:, :, :2] + 50
        labels = torch.randint(1, 151, (batch_size, max_objects))
        relation_pairs = torch.randint(0, max_objects, (batch_size, max_relations, 2))
        relation_labels = torch.randint(1, 51, (batch_size, max_relations))
        relation_mask = torch.ones(batch_size, max_relations, dtype=torch.bool)

        outputs = model(
            obj_features, boxes, labels, relation_pairs,
            relation_mask=relation_mask
        )

        targets = {
            'relation_labels': relation_labels,
            'labels': labels
        }

        losses = model.compute_loss(outputs, targets, relation_mask)

        assert 'pred_loss' in losses
        assert 'total_loss' in losses
        assert losses['pred_loss'].item() >= 0


class TestSimpleVRDModel:
    """Tests for SimpleVRDModel."""

    def test_simple_model_creation(self):
        """Test simple model creation."""
        model = SimpleVRDModel(
            num_obj_classes=151,
            num_pred_classes=51
        )
        assert model is not None

    def test_simple_model_forward(self):
        """Test simple model forward pass."""
        model = SimpleVRDModel(
            num_obj_classes=151,
            num_pred_classes=51,
            embedding_dim=256
        )

        num_objects = 5
        labels = torch.randint(1, 151, (num_objects,))
        boxes = torch.randn(num_objects, 4).abs() * 100
        boxes[:, 2:] = boxes[:, :2] + 50
        relation_pairs = torch.tensor([
            [0, 1], [1, 0], [2, 3], [3, 2]
        ])

        logits = model(labels, boxes, relation_pairs)

        assert logits.shape == (4, 51)


class TestModelFactory:
    """Tests for model factory function."""

    def test_create_model_from_config(self):
        """Test creating model from config dict."""
        config = {
            'mode': 'predcls',
            'dataset': {
                'num_obj_classes': 100,
                'num_pred_classes': 30
            },
            'model': {
                'feature_dim': 128,
                'hidden_dim': 256,
                'spatial_dim': 64,
                'use_bias': False,
                'dropout': 0.5
            }
        }

        model = create_model(config)

        assert model.num_obj_classes == 100
        assert model.num_pred_classes == 30
        assert model.feature_dim == 128


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
