"""Learnable fusion of visual relation logits and LLM commonsense prior logits.

Fusion formula:
    final_logits = visual_logits + alpha * (prior_logits / tau)

If prior_logits is None (prior disabled), returns visual_logits unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """Fuses visual predicate logits with a commonsense prior vector.

    Args:
        alpha_init: Initial value of alpha (prior weight). Must be > 0.
        tau_init:   Initial value of tau (prior temperature). Must be > 0.
        learnable:  If True, alpha and tau are optimised during training.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        tau_init: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self._log_alpha = nn.Parameter(torch.tensor(float(alpha_init)).log())
            self._log_tau   = nn.Parameter(torch.tensor(float(tau_init)))
        else:
            self.register_buffer("_fixed_alpha", torch.tensor(float(alpha_init)))
            self.register_buffer("_fixed_tau",   torch.tensor(float(tau_init)))

    @property
    def alpha(self) -> torch.Tensor:
        if self.learnable:
            return self._log_alpha.exp()
        return self._fixed_alpha  # type: ignore[return-value]

    @property
    def tau(self) -> torch.Tensor:
        if self.learnable:
            return F.softplus(self._log_tau) + 1e-6
        return self._fixed_tau  # type: ignore[return-value]

    def forward(self, visual_logits: torch.Tensor, prior_logits: Optional[torch.Tensor]) -> torch.Tensor:
        if prior_logits is None:
            return visual_logits
        return visual_logits + self.alpha * (prior_logits / self.tau)
