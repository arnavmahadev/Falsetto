"""FXencoder-Segment Stage-1 detector (TASKS.md T-19).

Input is a short sequence of frozen FXencoder embeddings. A **self-attention**
Transformer encoder processes them, followed by masked **mean pooling** and a
1-logit head.

Gotcha (why self-attn, not cross-attn): the FXencoder is frozen, so there is no
learnable query to justify a cross-attention decoder as in AudioCAT — the
sequence attends to itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FXSegment(nn.Module):
    """Self-attention encoder over FXencoder embeddings -> 1 logit."""

    def __init__(
        self,
        encoder_dim: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(encoder_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        feats: torch.Tensor,  # [B, T, D_enc]
        key_padding_mask: torch.Tensor | None = None,  # [B, T], True=pad
        return_embedding: bool = False,
    ):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        x = self.input_proj(feats)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        embedding = _masked_mean(x, key_padding_mask)
        logits = self.head(embedding).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits


def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the time axis, ignoring padded positions (``True`` = pad)."""
    if key_padding_mask is None:
        return x.mean(dim=1)
    keep = (~key_padding_mask).unsqueeze(-1).to(x.dtype)  # [B, T, 1]
    summed = (x * keep).sum(dim=1)
    count = keep.sum(dim=1).clamp_min(1.0)
    return summed / count
