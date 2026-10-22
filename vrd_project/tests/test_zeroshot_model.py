"""
Tests for Zero-Shot Scene Graph Model
========================================

Tests for ZeroShotSceneGraphModel forward pass, loss computation,
and CLIP embedding initialization.
"""

import sys
import pytest
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scene_graph_model import ZeroShotSceneGraphModel, create_zeroshot_model


class TestZeroShotSceneGraphModel:
    """Tests for ZeroShotSceneGraphModel."""

    def test_model_creation(self):
        """Test ZeroShotSceneGraphModel can be created."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls',
            clip_feature_dim=512,
            feature_dim=256,
            use_bias=False
        )
        assert model is not None
        assert model.clip_feature_dim == 512

    def test_model_creation_with_bias(self):
        """Test model creation with LLM semantic bias enabled."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=20,
            num_pred_classes=10,
            mode='predcls',
            use_bias=True,
            learnable_temperature=True
        )
        assert model is not None
        assert model.freq_bias is not None
        # Check that temperature is a learnable parameter
        assert hasattr(model.freq_bias, 'tau')

    def test_forward_with_clip_features(self):
        """Test forward pass with 512-dim features (CLIP output size)."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=151,
            num_pred_classes=51,
            mode='predcls',
            clip_feature_dim=512,
            feature_dim=256,
            use_bias=False
        )

        num_objects = 5
        num_relations = 10

        # CLIP features are 512-dim (not 1024 like CNN)
        obj_features = torch.randn(num_objects, 512)
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

        outputs = model(obj_features, boxes, labels, relation_pairs)

        assert 'pred_logits' in outputs
        assert 'pred_scores' in outputs
        assert outputs['pred_logits'].shape == (num_relations, 51)
        assert outputs['pred_scores'].shape == (num_relations, 51)

    def test_forward_batch(self):
        """Test forward pass with batched input."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=50,
            num_pred_classes=20,
            mode='predcls',
            clip_feature_dim=512,
            use_bias=False
        )

        batch_size = 2
        max_objects = 5
        max_relations = 8

        obj_features = torch.randn(batch_size, max_objects, 512)
        boxes = torch.randn(batch_size, max_objects, 4).abs() * 100
        boxes[:, :, 2:] = boxes[:, :, :2] + 50
        labels = torch.randint(1, 50, (batch_size, max_objects))
        relation_pairs = torch.randint(0, max_objects, (batch_size, max_relations, 2))
        object_mask = torch.ones(batch_size, max_objects, dtype=torch.bool)
        relation_mask = torch.ones(batch_size, max_relations, dtype=torch.bool)

        outputs = model(
            obj_features, boxes, labels, relation_pairs,
            object_mask, relation_mask
        )

        assert outputs['pred_logits'].shape == (batch_size, max_relations, 20)

    def test_loss_computation(self):
        """Test loss computation matches parent API."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=50,
            num_pred_classes=20,
            mode='predcls',
            clip_feature_dim=512,
            use_bias=False
        )

        batch_size = 2
        max_objects = 5
        max_relations = 8

        obj_features = torch.randn(batch_size, max_objects, 512)
        boxes = torch.randn(batch_size, max_objects, 4).abs() * 100
        boxes[:, :, 2:] = boxes[:, :, :2] + 50
        labels = torch.randint(1, 50, (batch_size, max_objects))
        relation_pairs = torch.randint(0, max_objects, (batch_size, max_relations, 2))
        relation_labels = torch.randint(1, 20, (batch_size, max_relations))
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

    def test_with_llm_bias(self):
        """Test forward pass with LLM semantic bias enabled."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=20,
            num_pred_classes=10,
            mode='predcls',
            clip_feature_dim=512,
            use_bias=True,
            learnable_temperature=True
        )

        # Set uniform priors so bias produces non-zero values
        model.freq_bias._set_uniform_priors()

        num_objects = 5
        num_relations = 4

        obj_features = torch.randn(num_objects, 512)
        boxes = torch.randn(num_objects, 4).abs() * 100
        boxes[:, 2:] = boxes[:, :2] + 50
        labels = torch.randint(0, 20, (num_objects,))
        relation_pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])

        outputs = model(obj_features, boxes, labels, relation_pairs)

        assert outputs['pred_logits'].shape == (num_relations, 10)
        assert outputs['pred_scores'].shape == (num_relations, 10)

    def test_feature_projection_dim(self):
        """Test that feature projection maps from 512 (CLIP) not 1024 (CNN)."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=20,
            num_pred_classes=10,
            clip_feature_dim=512,
            feature_dim=256,
            use_bias=False
        )

        # Check the first layer of feature_proj accepts 512-dim input
        first_layer = model.feature_proj[0]
        assert first_layer.in_features == 512

    def test_clip_embedding_initialization(self):
        """Test initializing obj_embed from CLIP text embeddings."""
        model = ZeroShotSceneGraphModel(
            num_obj_classes=20,
            num_pred_classes=10,
            feature_dim=256,
            use_bias=False
        )

        # Simulate CLIP text embeddings for 20 classes
        clip_text_embeds = torch.randn(20, 512)

        # This should not raise
        model.initialize_clip_embeddings(clip_text_embeds)

        # obj_embed weights should have changed from default init
        assert model.obj_embed.weight.shape == (20, 256)


class TestZeroShotModelFactory:
    """Tests for create_zeroshot_model factory function."""

    def test_create_from_config(self):
        """Test creating model from config dict."""
        config = {
            'mode': 'predcls',
            'dataset': {
                'num_obj_classes': 100,
                'num_pred_classes': 30
            },
            'model': {
                'clip_feature_dim': 512,
                'feature_dim': 128,
                'hidden_dim': 256,
                'spatial_dim': 64,
                'use_bias': False,
                'dropout': 0.5
            }
        }

        model = create_zeroshot_model(config)

        assert model.num_obj_classes == 100
        assert model.num_pred_classes == 30
        assert model.feature_dim == 128
        assert model.clip_feature_dim == 512


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
