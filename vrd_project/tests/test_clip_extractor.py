"""
Tests for CLIP Feature Extractor
==================================

Tests for CLIPFeatureExtractor module used in zero-shot VRD.
"""

import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_feature_extractor import CLIPFeatureExtractor


class TestCLIPFeatureExtractor:
    """Tests for CLIPFeatureExtractor module."""

    def test_creation(self):
        """Test CLIPFeatureExtractor can be created without loading model."""
        extractor = CLIPFeatureExtractor(
            clip_model_name='ViT-B-32',
            pretrained='openai'
        )
        assert extractor is not None
        assert extractor.feature_dim == 512
        assert extractor._model is None  # Lazy loading

    def test_get_feature_dim(self):
        """Test feature dimension is correct."""
        extractor = CLIPFeatureExtractor()
        assert extractor.get_feature_dim() == 512

    def test_dummy_features(self):
        """Test dummy feature generation (no CLIP model needed)."""
        extractor = CLIPFeatureExtractor()

        num_objects = 10
        device = torch.device('cpu')
        features = extractor.extract_dummy_features(num_objects, device)

        assert features.shape == (10, 512)
        assert (features == 0).all()

    def test_forward_raises(self):
        """Test that calling forward() directly raises NotImplementedError."""
        extractor = CLIPFeatureExtractor()
        with pytest.raises(NotImplementedError):
            extractor()

    def test_factory_function(self):
        """Test create_clip_extractor factory."""
        from src.models.clip_feature_extractor import create_clip_extractor

        config = {
            'model': {
                'clip_model': 'ViT-B-32',
                'clip_pretrained': 'openai'
            },
            'device': 'cpu'
        }

        extractor = create_clip_extractor(config)
        assert extractor is not None
        assert extractor.clip_model_name == 'ViT-B-32'
        assert extractor.pretrained == 'openai'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
