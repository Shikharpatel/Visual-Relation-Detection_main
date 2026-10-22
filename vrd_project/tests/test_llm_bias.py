"""
Tests for LLM Semantic Bias Module
====================================

Tests for LLMSemanticBias module used in zero-shot VRD.
"""

import sys
import pytest
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.llm_semantic_bias import LLMSemanticBias


class TestLLMSemanticBias:
    """Tests for LLMSemanticBias module."""

    def test_creation(self):
        """Test LLMSemanticBias can be created."""
        bias = LLMSemanticBias(
            num_obj_classes=151,
            num_pred_classes=51,
            alpha=1.0
        )
        assert bias is not None
        assert not bias.is_loaded

    def test_forward_unloaded(self):
        """Test forward returns zeros when priors not loaded."""
        bias = LLMSemanticBias(
            num_obj_classes=151,
            num_pred_classes=51
        )

        subj_labels = torch.tensor([1, 2, 3])
        obj_labels = torch.tensor([4, 5, 6])

        logits = bias(subj_labels, obj_labels)

        assert logits.shape == (3, 51)
        assert (logits == 0).all()

    def test_forward_with_uniform_priors(self):
        """Test forward with uniform priors."""
        bias = LLMSemanticBias(
            num_obj_classes=10,
            num_pred_classes=5,
            alpha=1.0
        )

        # Set uniform priors
        bias._set_uniform_priors()
        assert bias.is_loaded

        subj_labels = torch.tensor([1, 2, 3])
        obj_labels = torch.tensor([4, 5, 6])

        logits = bias(subj_labels, obj_labels)

        assert logits.shape == (3, 5)
        assert not (logits == 0).all()

    def test_load_from_tensor(self):
        """Test loading priors from a tensor."""
        bias = LLMSemanticBias(
            num_obj_classes=10,
            num_pred_classes=5
        )

        prior_tensor = torch.rand(10, 10, 5)
        # Normalize
        prior_tensor = prior_tensor / prior_tensor.sum(dim=-1, keepdim=True)

        bias.load_from_tensor(prior_tensor)

        assert bias.is_loaded
        assert torch.allclose(bias.prior_matrix, prior_tensor)

    def test_learnable_temperature(self):
        """Test that temperature parameter is learnable."""
        bias = LLMSemanticBias(
            num_obj_classes=10,
            num_pred_classes=5,
            learnable_temperature=True,
            initial_temperature=1.0
        )

        assert isinstance(bias.tau, torch.nn.Parameter)
        assert bias.tau.requires_grad
        assert bias.tau.item() == 1.0

    def test_fixed_temperature(self):
        """Test fixed (non-learnable) temperature."""
        bias = LLMSemanticBias(
            num_obj_classes=10,
            num_pred_classes=5,
            learnable_temperature=False,
            initial_temperature=2.0
        )

        assert not isinstance(bias.tau, torch.nn.Parameter)
        assert bias.tau.item() == 2.0

    def test_get_top_predicates(self):
        """Test getting top predicates for a pair."""
        bias = LLMSemanticBias(
            num_obj_classes=10,
            num_pred_classes=5
        )

        # Set custom priors for a specific pair
        bias.prior_matrix[1, 2] = torch.tensor([0.1, 0.5, 0.2, 0.15, 0.05])
        bias.is_loaded.fill_(True)

        top_preds = bias.get_top_predicates(1, 2, top_k=3)

        assert len(top_preds) == 3
        # First should be index 1 (highest prob 0.5)
        assert top_preds[0][0] == 1
        assert abs(top_preds[0][1] - 0.5) < 1e-5

    def test_api_compatibility_with_frequency_bias(self):
        """Test that forward() has same API and output shape as FrequencyBias."""
        from src.models.frequency_bias import FrequencyBias

        num_obj = 20
        num_pred = 10
        N = 5

        freq_bias = FrequencyBias(num_obj, num_pred)
        llm_bias = LLMSemanticBias(num_obj, num_pred)

        subj_labels = torch.randint(0, num_obj, (N,))
        obj_labels = torch.randint(0, num_obj, (N,))

        freq_output = freq_bias(subj_labels, obj_labels)
        llm_output = llm_bias(subj_labels, obj_labels)

        # Same shapes
        assert freq_output.shape == llm_output.shape
        assert freq_output.shape == (N, num_pred)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading bias."""
        bias = LLMSemanticBias(num_obj_classes=10, num_pred_classes=5)
        bias._set_uniform_priors()

        save_path = str(tmp_path / 'test_bias.pt')
        bias.save(save_path)

        bias2 = LLMSemanticBias(num_obj_classes=10, num_pred_classes=5)
        bias2.load(save_path)

        assert bias2.is_loaded
        assert torch.allclose(bias.prior_matrix, bias2.prior_matrix)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
