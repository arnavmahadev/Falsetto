"""Embedding stream (TASKS.md T-30, Paper 2).

Linear projection + positional encoding + Transformer encoder (x6) over the
segment-embedding sequence ``E`` -> ``X_EMB``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common import PositionalEncoding, transformer_encoder


class EmbeddingStream(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder = transformer_encoder(d_model, n_heads, ffn_dim, n_layers, dropout)

    def forward(self, embeddings: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos(self.proj(embeddings))  # [B, N, d_model]
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
