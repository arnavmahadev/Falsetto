"""Gated Multimodal Unit fusion (TASKS.md T-33, Paper 2).

    G       = sigmoid(W_g · [X_contents ; X_structure] + b_g)
    X_fused = G ⊙ X_contents + (1 - G) ⊙ X_structure

The per-segment gate ``G`` is returned so Phase 6 can visualize how much each
segment weights content vs. structure (real vs. fake; Fig. 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedMultimodalUnit(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(
        self, x_contents: torch.Tensor, x_structure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(X_fused [B, N, d], G [B, N, d])``."""
        g = torch.sigmoid(self.gate(torch.cat([x_contents, x_structure], dim=-1)))
        x_fused = g * x_contents + (1.0 - g) * x_structure
        return x_fused, g
