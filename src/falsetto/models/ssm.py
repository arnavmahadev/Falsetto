"""Self-similarity matrix over segment embeddings (TASKS.md T-27).

Paper 2 defines ``SSM[i, j] = exp(-||e_i - e_j||^2 / d)`` where ``d`` is the
embedding dimension. The matrix is symmetric with a ~1 diagonal (distance 0),
and captures how a track's segments repeat and vary — the structural signal
Stage-2 reasons over. This exact form is reused by the Fusion model (Phase 5).
"""

from __future__ import annotations

import torch


def self_similarity_matrix(
    embeddings: torch.Tensor,
    scale: float | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``exp(-||e_i - e_j||^2 / d)``.

    Args:
        embeddings: ``[N, D]`` or ``[B, N, D]``.
        scale: denominator ``d`` (defaults to the embedding dim ``D``).
        key_padding_mask: ``[N]`` / ``[B, N]`` bool, ``True`` = padded position;
            padded rows/cols are zeroed out.

    Returns:
        ``[N, N]`` or ``[B, N, N]`` similarity matrix.
    """
    single = embeddings.dim() == 2
    if single:
        embeddings = embeddings.unsqueeze(0)
    b, n, d = embeddings.shape
    denom = float(scale) if scale is not None else float(d)

    # Pairwise squared Euclidean distance, numerically stable via cdist.
    dist = torch.cdist(embeddings, embeddings, p=2) ** 2  # [B, N, N]
    ssm = torch.exp(-dist / denom)

    if key_padding_mask is not None:
        if key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        keep = (~key_padding_mask).to(ssm.dtype)  # [B, N]
        ssm = ssm * keep.unsqueeze(1) * keep.unsqueeze(2)

    return ssm.squeeze(0) if single else ssm
