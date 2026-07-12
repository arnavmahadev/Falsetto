"""Fusion Segment Transformer (TASKS.md T-34, Paper 2 — headline model).

Wires the two streams (T-30/T-31), bi-directional cross-attention (T-32) and the
gated fusion (T-33) into a Stage-2 classifier:

    E, SSM -> [EmbeddingStream, SSMStream] -> X_EMB, X_SSM
           -> BiDirectionalCrossAttention  -> X_contents, X_structure
           -> GatedMultimodalUnit          -> X_fused, G
           -> masked mean -> head          -> logit

``G`` (the per-segment content/structure gate) is exposed for Phase-6 analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common import masked_mean
from ..ssm import self_similarity_matrix
from .cross_attention import BiDirectionalCrossAttention
from .embedding_stream import EmbeddingStream
from .gate import GatedMultimodalUnit
from .ssm_stream import SSMStream


class FusionSegmentTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        emb_stream_layers: int = 6,
        ssm_stream_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 1,
        fusion: str = "gmu",  # "gmu" (gated) | "mean" (plain cross-attn, ablation T-38)
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fusion = fusion.lower()
        self.embedding_stream = EmbeddingStream(
            embed_dim, seq_len, d_model, n_heads, ffn_dim, emb_stream_layers, dropout
        )
        self.ssm_stream = SSMStream(seq_len, d_model, n_heads, ffn_dim, ssm_stream_layers, dropout)
        self.cross_attention = BiDirectionalCrossAttention(d_model, n_heads, dropout)
        self.gate = GatedMultimodalUnit(d_model) if self.fusion == "gmu" else None
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, N, D]
        ssm: torch.Tensor | None = None,  # [B, N, N]
        key_padding_mask: torch.Tensor | None = None,  # [B, N], True=pad
        return_gate: bool = False,
        return_embedding: bool = False,
    ):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if ssm is None:
            ssm = self_similarity_matrix(embeddings, key_padding_mask=key_padding_mask)
        if ssm.dim() == 2:
            ssm = ssm.unsqueeze(0)

        x_emb = self.embedding_stream(embeddings, key_padding_mask)
        x_ssm = self.ssm_stream(ssm, key_padding_mask)
        x_contents, x_structure = self.cross_attention(x_emb, x_ssm, key_padding_mask)
        if self.gate is not None:
            x_fused, g = self.gate(x_contents, x_structure)
        else:  # plain cross-attention fusion (ablation): unweighted average
            x_fused = 0.5 * (x_contents + x_structure)
            g = None

        pooled = masked_mean(self.norm(x_fused), key_padding_mask)
        logits = self.head(pooled).squeeze(-1)

        if return_gate and return_embedding:
            return logits, g, pooled
        if return_gate:
            return logits, g
        if return_embedding:
            return logits, pooled
        return logits
