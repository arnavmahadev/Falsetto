"""Stage-1 losses (TASKS.md T-20).

- :class:`BCEWithLogits` — the AudioCAT path (thin wrapper over the torch loss,
  accepting ``[B]`` or ``[B, 1]`` logits/targets).
- :class:`FocalLoss` — the FXencoder path; down-weights easy examples via
  ``(1 - p_t)^gamma`` with an ``alpha`` class weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits.reshape(-1), targets.reshape(-1).to(logits.dtype)


class BCEWithLogits(nn.Module):
    """Binary cross-entropy on raw logits (real=0 / ai=1)."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _flatten(logits, targets)
        pos_weight = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) on logits."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _flatten(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # prob of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, **kwargs) -> nn.Module:
    """Factory: ``"bce"`` or ``"focal"``."""
    name = name.lower()
    if name == "bce":
        return BCEWithLogits(pos_weight=kwargs.get("pos_weight"))
    if name == "focal":
        return FocalLoss(
            gamma=kwargs.get("focal_gamma", 2.0),
            alpha=kwargs.get("focal_alpha", 0.25),
        )
    raise ValueError(f"unknown loss {name!r} (expected 'bce' or 'focal')")
