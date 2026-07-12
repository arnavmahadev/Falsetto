"""SSM stream (TASKS.md T-31, Paper 2).

Conv1d projection of the SSM rows + positional encoding + Transformer encoder
(x2) -> ``X_SSM``. The SSM ``[B, N, N]`` is treated as ``N`` feature channels over
``N`` sequence positions; a length-3 Conv1d projects each position's row context
to ``d_model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common import PositionalEncoding, transformer_encoder


class SSMStream(nn.Module):
    def __init__(
        self,
        seq_len: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # in_channels = N (the row/similarity dimension), out = d_model, over N positions.
        self.proj = nn.Conv1d(seq_len, d_model, kernel_size=3, padding=1)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder = transformer_encoder(d_model, n_heads, ffn_dim, n_layers, dropout)

    def forward(self, ssm: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ssm [B, N, N] -> [B, C=N, L=N] for Conv1d over positions.
        x = ssm.transpose(1, 2)
        x = self.proj(x).transpose(1, 2)  # [B, N, d_model]
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
