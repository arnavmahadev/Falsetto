"""Segment-embedding sequence builder (TASKS.md T-26).

Turns a full track into the fixed-length sequence Stage-2 consumes:

    track -> beat-track -> 4-bar segments -> Stage-1 embed each -> E=[e_1..e_N]
          -> pad/crop to N=48 (+ padding mask)

The Stage-1 embedder is injected as a callable (``embed_fn: waveform -> [D]``) so
this stays decoupled from any particular head; :func:`stage1_embedding_fn` wraps a
trained :class:`~falsetto.models.stage1.Stage1Detector` to produce one.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..utils.audio_io import resample
from .beat import BeatTracker
from .segment_bars import four_bar_segments

EmbedFn = Callable[[torch.Tensor], torch.Tensor]


def pad_crop_sequence(embeddings: torch.Tensor, n: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/crop ``[M, D]`` to ``[n, D]`` and return ``(E, mask)`` (mask True = pad)."""
    m, d = embeddings.shape
    mask = torch.ones(n, dtype=torch.bool, device=embeddings.device)
    out = embeddings.new_zeros(n, d)
    keep = min(m, n)
    out[:keep] = embeddings[:keep]
    mask[:keep] = False
    return out, mask


class SegmentSequenceBuilder:
    """Build ``([48, D], mask)`` from a track waveform."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        sample_rate: int,
        beat_tracker: BeatTracker | None = None,
        max_segments: int = 48,
        bars_per_segment: int = 4,
        quantize_seconds: float | None = None,
        fallback_clip_seconds: float = 10.0,
        segmentation: str = "downbeat",  # "downbeat" (4-bar) | "fixed" (window; ablation T-37)
    ) -> None:
        self.embed_fn = embed_fn
        self.sample_rate = sample_rate
        self.beat_tracker = beat_tracker or BeatTracker()
        self.max_segments = max_segments
        self.bars_per_segment = bars_per_segment
        self.quantize_seconds = quantize_seconds
        self.fallback_clip_seconds = fallback_clip_seconds
        self.segmentation = segmentation

    def build(self, waveform: torch.Tensor, sr_in: int) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr_in != self.sample_rate:
            waveform = resample(waveform, sr_in, self.sample_rate)

        # "fixed" ablation skips beat tracking -> four_bar_segments falls back to windows.
        if self.segmentation == "fixed":
            downbeats: list[float] = []
        else:
            downbeats = self.beat_tracker.track(waveform, self.sample_rate).downbeats
        seg = four_bar_segments(
            waveform,
            self.sample_rate,
            downbeats,
            bars_per_segment=self.bars_per_segment,
            quantize_seconds=self.quantize_seconds,
            fallback_clip_seconds=self.fallback_clip_seconds,
        )
        embeddings = torch.stack([self.embed_fn(s) for s in seg.segments], dim=0)  # [M, D]
        return pad_crop_sequence(embeddings, self.max_segments)


def stage1_embedding_fn(model, device: torch.device | None = None) -> EmbedFn:
    """Wrap a Stage1Detector so a single segment waveform -> its Segment F.E Embedding ``[D]``."""
    model.eval()

    @torch.no_grad()
    def embed(segment: torch.Tensor) -> torch.Tensor:
        if device is not None:
            segment = segment.to(device)
        _logit, emb = model(segment, return_embedding=True)  # emb [1, d_model]
        return emb.squeeze(0)

    return embed
