"""Bi-directional cross-attention (TASKS.md T-32, Paper 2).

Each stream attends over the other, with a residual + LayerNorm:

    X_contents  = LayerNorm(X_EMB + CrossAttn(Q=X_EMB, K=V=X_SSM))
    X_structure = LayerNorm(X_SSM + CrossAttn(Q=X_SSM, K=V=X_EMB))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb_to_ssm = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ssm_to_emb = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_contents = nn.LayerNorm(d_model)
        self.norm_structure = nn.LayerNorm(d_model)

    def forward(
        self,
        x_emb: torch.Tensor,  # [B, N, d]
        x_ssm: torch.Tensor,  # [B, N, d]
        key_padding_mask: torch.Tensor | None = None,  # [B, N], True=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Content = embeddings enriched by structure.
        attn_c, _ = self.emb_to_ssm(
            x_emb, x_ssm, x_ssm, key_padding_mask=key_padding_mask, need_weights=False
        )
        x_contents = self.norm_contents(x_emb + attn_c)

        # Structure = SSM enriched by content.
        attn_s, _ = self.ssm_to_emb(
            x_ssm, x_emb, x_emb, key_padding_mask=key_padding_mask, need_weights=False
        )
        x_structure = self.norm_structure(x_ssm + attn_s)
        return x_contents, x_structure
