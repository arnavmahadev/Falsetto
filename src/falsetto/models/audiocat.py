"""AudioCAT — cross-attention Stage-1 detector (TASKS.md T-18).

A small set of **learnable latent query tokens** attend over an extractor's
feature map through a stack of Transformer **decoder** blocks:

    for each of n_layers:
        latents = latents + SelfAttn(latents)                 # latents talk to latents
        latents = latents + CrossAttn(Q=latents, K=V=feats)   # latents read the encoder
        latents = latents + FFN(latents)                      # (pre-norm residuals)

The pooled latents form the **Segment F.E Embedding**, fed to a 1-logit head.
The encoder is any Phase-2 extractor (swapped via config); AudioCAT only sees its
``[B, T, D_enc]`` feature map plus an optional key-padding mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLayerSum(nn.Module):
    """Learnable softmax-weighted sum over an extractor's hidden-state layers.

    Consumes ``[B, L, T, D]`` (or ``[L, T, D]``) and returns ``[B, T, D]`` — the
    ``layer_strategy="weighted"`` path for MERT/Music2Vec.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        if layers.dim() == 3:  # [L, T, D] -> [1, L, T, D]
            layers = layers.unsqueeze(0)
        w = F.softmax(self.weights, dim=0).view(1, -1, 1, 1)
        return (layers * w).sum(dim=1)


class _CrossAttnDecoderLayer(nn.Module):
    """Pre-norm decoder block: self-attn(latents) + cross-attn(latents, feats) + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,  # [B, Q, d]
        memory: torch.Tensor,  # [B, T, d]
        memory_key_padding_mask: torch.Tensor | None = None,  # [B, T], True=pad
    ) -> torch.Tensor:
        q = self.norm_sa(latents)
        sa, _ = self.self_attn(q, q, q, need_weights=False)
        latents = latents + self.dropout(sa)

        q = self.norm_ca(latents)
        ca, _ = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        latents = latents + self.dropout(ca)

        latents = latents + self.ffn(self.norm_ff(latents))
        return latents


class AudioCAT(nn.Module):
    """Cross-attention decoder head over extractor features -> 1 logit."""

    def __init__(
        self,
        encoder_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_latents: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(encoder_dim, d_model)
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_CrossAttnDecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(
        self,
        feats: torch.Tensor,  # [B, T, D_enc]
        key_padding_mask: torch.Tensor | None = None,  # [B, T], True=pad
        return_embedding: bool = False,
    ):
        if feats.dim() == 2:  # [T, D] -> [1, T, D]
            feats = feats.unsqueeze(0)
        memory = self.input_proj(feats)
        latents = self.latents.unsqueeze(0).expand(memory.size(0), -1, -1)
        for layer in self.layers:
            latents = layer(latents, memory, key_padding_mask)
        embedding = self.norm(latents).mean(dim=1)  # [B, d_model] — Segment F.E Embedding
        logits = self.head(embedding).squeeze(-1)  # [B] when num_classes==1
        if return_embedding:
            return logits, embedding
        return logits
