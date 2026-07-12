"""Shared model building blocks (positional encoding, pooling, encoder stacks)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding, added to ``[B, N, d]``."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the sequence axis ignoring padded positions (``True`` = pad)."""
    if key_padding_mask is None:
        return x.mean(dim=1)
    keep = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
    return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


def transformer_encoder(
    d_model: int, n_heads: int, ffn_dim: int, n_layers: int, dropout: float
) -> nn.TransformerEncoder:
    """Pre-norm GELU Transformer encoder stack (batch-first)."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    # norm_first=True makes the nested-tensor fast path inapplicable; disable it
    # explicitly to avoid a noisy UserWarning.
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
