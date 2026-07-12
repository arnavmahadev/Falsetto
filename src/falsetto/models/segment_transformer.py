"""Segment Transformer — dual-pathway Stage-2 classifier (TASKS.md T-28, Paper 1).

Two parallel Transformer encoders reason over a track's segment sequence:

    Content pathway   : E   [B, N, D]  -> proj+pos -> encoder -> masked mean
    Structure pathway : SSM [B, N, N]  -> proj+pos -> encoder -> masked mean

Their pooled outputs are **concatenated** and passed to a 1-logit head. The SSM
is built from the segment embeddings (``exp(-||e_i-e_j||^2/d)``) if not supplied.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import PositionalEncoding, masked_mean, transformer_encoder
from .ssm import self_similarity_matrix


class SegmentTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # Content pathway
        self.content_proj = nn.Linear(embed_dim, d_model)
        self.content_pos = PositionalEncoding(d_model, max_len=seq_len)
        self.content_encoder = transformer_encoder(d_model, n_heads, ffn_dim, n_layers, dropout)

        # Structure pathway (each SSM row is a length-N feature vector)
        self.structure_proj = nn.Linear(seq_len, d_model)
        self.structure_pos = PositionalEncoding(d_model, max_len=seq_len)
        self.structure_encoder = transformer_encoder(d_model, n_heads, ffn_dim, n_layers, dropout)

        self.norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Linear(2 * d_model, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, N, D]
        ssm: torch.Tensor | None = None,  # [B, N, N]
        key_padding_mask: torch.Tensor | None = None,  # [B, N], True=pad
        return_embedding: bool = False,
    ):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if ssm is None:
            ssm = self_similarity_matrix(embeddings, key_padding_mask=key_padding_mask)
        if ssm.dim() == 2:
            ssm = ssm.unsqueeze(0)

        c = self.content_encoder(
            self.content_pos(self.content_proj(embeddings)), src_key_padding_mask=key_padding_mask
        )
        content_vec = masked_mean(c, key_padding_mask)

        s = self.structure_encoder(
            self.structure_pos(self.structure_proj(ssm)), src_key_padding_mask=key_padding_mask
        )
        structure_vec = masked_mean(s, key_padding_mask)

        fused = self.norm(torch.cat([content_vec, structure_vec], dim=-1))
        logits = self.head(fused).squeeze(-1)
        if return_embedding:
            return logits, fused
        return logits
