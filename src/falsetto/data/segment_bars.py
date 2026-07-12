"""4-bar (downbeat) segmentation (TASKS.md T-25).

Group detected downbeats into **4-bar units**, slice the audio at those
boundaries, and optionally quantize every segment to a uniform temporal length.
When downbeats are too sparse (free tempo, tracking failure), fall back to fixed
windows so the pipeline degrades gracefully instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .segment_fixed import _pad_to, segment_fixed


@dataclass
class BarSegmentation:
    segments: list[torch.Tensor]  # each [channels, samples]
    boundaries_sec: list[tuple[float, float]]
    method: str  # "downbeat" | "fixed_fallback"


def four_bar_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    downbeats: list[float],
    bars_per_segment: int = 4,
    quantize_seconds: float | None = None,
    fallback_clip_seconds: float = 10.0,
) -> BarSegmentation:
    """Slice ``waveform`` into 4-bar segments on downbeat boundaries.

    Args:
        waveform: ``[channels, samples]`` or ``[samples]``.
        sample_rate: sample rate.
        downbeats: downbeat timestamps in seconds (from :mod:`falsetto.data.beat`).
        bars_per_segment: downbeats grouped per segment (4 -> 4-bar units).
        quantize_seconds: if set, pad/crop every segment to this length (uniform grid).
        fallback_clip_seconds: window length when downbeats are insufficient.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    total = waveform.size(1)

    # Need at least two segment boundaries to cut one segment.
    if len(downbeats) < bars_per_segment + 1:
        return _fixed_fallback(waveform, sample_rate, fallback_clip_seconds, quantize_seconds)

    # Boundaries every `bars_per_segment` downbeats.
    boundary_idx = list(range(0, len(downbeats), bars_per_segment))
    boundary_times = [downbeats[i] for i in boundary_idx]
    if boundary_times[-1] < downbeats[-1]:
        boundary_times.append(downbeats[-1])  # close the final segment

    segments: list[torch.Tensor] = []
    spans: list[tuple[float, float]] = []
    for start_t, end_t in zip(boundary_times[:-1], boundary_times[1:], strict=False):
        s = max(0, int(round(start_t * sample_rate)))
        e = min(total, int(round(end_t * sample_rate)))
        if e - s < int(0.05 * sample_rate):  # skip <50 ms slivers
            continue
        seg = waveform[:, s:e]
        if quantize_seconds is not None:
            seg = _pad_to(seg, int(round(quantize_seconds * sample_rate)))
        segments.append(seg.contiguous())
        spans.append((start_t, end_t))

    if not segments:
        return _fixed_fallback(waveform, sample_rate, fallback_clip_seconds, quantize_seconds)
    return BarSegmentation(segments, spans, method="downbeat")


def _fixed_fallback(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    quantize_seconds: float | None,
) -> BarSegmentation:
    clips = segment_fixed(waveform, sample_rate, clip_seconds)  # [N, C, S]
    segments, spans = [], []
    for i in range(clips.size(0)):
        seg = clips[i]
        if quantize_seconds is not None:
            seg = _pad_to(seg, int(round(quantize_seconds * sample_rate)))
        segments.append(seg.contiguous())
        spans.append((i * clip_seconds, (i + 1) * clip_seconds))
    return BarSegmentation(segments, spans, method="fixed_fallback")
