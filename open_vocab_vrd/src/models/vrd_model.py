"""Top-level model that composes RelationHead + FusionModule.

CLIPEncoder and YOLOWorldDetector are NOT included here — they are
used outside the training graph (frozen, pre-extracted features).

During training:
    model(subj_emb, obj_emb, spatial, prior_vec) -> logits

During inference:
    See scripts/infer.py for the full detect -> encode -> pair -> predict pipeline.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.fusion import FusionModule
from src.models.relation_head import RelationHead, build_relation_head


class OpenVocabVRDModel(nn.Module):
    """Predicate prediction model (trainable part of the pipeline).

    Args:
        relation_head: RelationHead instance.
        fusion:        FusionModule instance.
    """

    def __init__(self, relation_head: RelationHead, fusion: FusionModule):
        super().__init__()
        self.relation_head = relation_head
        self.fusion = fusion

    def forward(
        self,
        subj_emb: torch.Tensor,
        obj_emb: torch.Tensor,
        spatial: torch.Tensor,
        prior_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict predicate logits for a batch of subject-object pairs.

        Args:
            subj_emb:  (B, 512) CLIP subject embeddings.
            obj_emb:   (B, 512) CLIP object embeddings.
            spatial:   (B, 13)  spatial feature vectors.
            prior_vec: (B, num_predicates) prior logits, or None.

        Returns:
            (B, num_predicates) logits (pre-softmax).
        """
        visual_logits = self.relation_head(subj_emb, obj_emb, spatial)
        return self.fusion(visual_logits, prior_vec)

    @torch.no_grad()
    def predict(
        self,
        subj_emb: torch.Tensor,
        obj_emb: torch.Tensor,
        spatial: torch.Tensor,
        prior_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference convenience: returns softmax probabilities."""
        self.eval()
        logits = self.forward(subj_emb, obj_emb, spatial, prior_vec)
        return torch.softmax(logits, dim=-1)


def build_model(cfg: dict, num_predicates: int) -> OpenVocabVRDModel:
    """Construct model from config dict."""
    head = build_relation_head(cfg["relation_head"], num_predicates)
    prior_cfg = cfg.get("prior", {})
    fusion = FusionModule(
        alpha_init=prior_cfg.get("alpha_init", 1.0),
        tau_init=prior_cfg.get("tau_init", 1.0),
        learnable=prior_cfg.get("learnable", True),
    )
    return OpenVocabVRDModel(head, fusion)


def load_model(checkpoint_path: str, cfg: dict, num_predicates: int, device: torch.device) -> OpenVocabVRDModel:
    """Load a saved model checkpoint."""
    model = build_model(cfg, num_predicates)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
