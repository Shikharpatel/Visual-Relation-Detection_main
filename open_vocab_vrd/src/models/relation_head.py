"""Lightweight MLP relation head: predicts predicate logits from CLIP + spatial features.

Input  = concat(subj_emb[512], obj_emb[512], spatial[13]) = 1037-D
Output = logits over num_predicates classes

All dimensions are config-driven so the model can be extended without code changes.
"""

from typing import List

import torch
import torch.nn as nn


class RelationHead(nn.Module):
    """MLP predicate classifier.

    Args:
        input_dim:     Total input dimension (default 1037 = 512+512+13).
        hidden_dims:   List of hidden layer sizes (default [512, 256]).
        num_predicates: Number of output classes.
        dropout:       Dropout probability applied after each hidden layer.
        use_layer_norm: Apply LayerNorm on the input before the first linear.
    """

    def __init__(
        self,
        input_dim: int = 1037,
        hidden_dims: List[int] = None,
        num_predicates: int = 20,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: List[nn.Module] = []

        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_predicates))

        self.net = nn.Sequential(*layers)

    def forward(self, subj_emb: torch.Tensor, obj_emb: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subj_emb: (B, clip_dim) subject CLIP embeddings.
            obj_emb:  (B, clip_dim) object  CLIP embeddings.
            spatial:  (B, spatial_dim) spatial feature vectors.

        Returns:
            logits: (B, num_predicates)
        """
        x = torch.cat([subj_emb, obj_emb, spatial], dim=-1)
        return self.net(x)


def build_relation_head(cfg: dict, num_predicates: int) -> RelationHead:
    """Construct a RelationHead from a config dict (default.yaml["relation_head"])."""
    return RelationHead(
        input_dim=cfg.get("input_dim", 1037),
        hidden_dims=cfg.get("hidden_dims", [512, 256]),
        num_predicates=num_predicates,
        dropout=cfg.get("dropout", 0.3),
        use_layer_norm=cfg.get("use_layer_norm", True),
    )
