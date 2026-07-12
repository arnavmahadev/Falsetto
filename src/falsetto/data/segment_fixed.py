"""Stage-1 fixed-window segmentation (TASKS.md T-08).

Cut a track into fixed-length windows (10 s for FakeMusicCaps; 5 s / 10 s for
SONICS). Boundaries are deterministic (``floor``-aligned by hop). Short clips are
right-padded with zeros so every window has exactly ``clip_seconds`` of audio.
"""

from __future__ import annotations

import torch


def segment_fixed(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    hop_seconds: float | None = None,
    pad: bool = True,
    drop_last_partial: bool = False,
) -> torch.Tensor:
    """Split a waveform into fixed windows.

    Args:
        waveform: ``[samples]`` or ``[channels, samples]``.
        sample_rate: sample rate of ``waveform``.
        clip_seconds: window length in seconds.
        hop_seconds: stride between windows (defaults to ``clip_seconds`` — no overlap).
        pad: right-pad the final (or a too-short) window with zeros.
        drop_last_partial: if True, discard a trailing partial window instead of padding it.

    Returns:
        ``[num_clips, channels, clip_samples]``. For mono input, ``channels == 1``.
        Boundaries are deterministic given the arguments.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, samples]
    if waveform.dim() != 2:
        raise ValueError(f"expected [samples] or [channels, samples], got {tuple(waveform.shape)}")

    channels, total = waveform.shape
    clip_samples = int(round(clip_seconds * sample_rate))
    hop_samples = int(round((hop_seconds if hop_seconds is not None else clip_seconds) * sample_rate))
    if clip_samples <= 0 or hop_samples <= 0:
        raise ValueError("clip_seconds and hop_seconds must be positive")

    if total <= clip_samples:
        # Single window; pad up to clip length if needed.
        if total < clip_samples and pad:
            waveform = _pad_to(waveform, clip_samples)
        elif total < clip_samples and not pad:
            return waveform.unsqueeze(0)
        return waveform.unsqueeze(0)  # [1, channels, clip_samples]

    clips = []
    start = 0
    while start + clip_samples <= total:
        clips.append(waveform[:, start : start + clip_samples])
        start += hop_samples

    # Trailing remainder that a full-length window didn't cover.
    if start < total and not drop_last_partial:
        tail = waveform[:, start:]
        if pad:
            clips.append(_pad_to(tail, clip_samples))
        # if not padding, a shorter tail is dropped to keep a uniform stack

    return torch.stack(clips, dim=0)  # [num_clips, channels, clip_samples]


def _pad_to(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Right-pad ``[channels, samples]`` with zeros to ``length`` (or crop if longer)."""
    channels, samples = waveform.shape
    if samples == length:
        return waveform
    if samples > length:
        return waveform[:, :length]
    pad = waveform.new_zeros(channels, length - samples)
    return torch.cat([waveform, pad], dim=1)


def num_windows(total_samples: int, clip_samples: int, hop_samples: int, pad: bool = True) -> int:
    """How many windows :func:`segment_fixed` would produce (for preallocation)."""
    if total_samples <= clip_samples:
        return 1
    n = 1 + (total_samples - clip_samples) // hop_samples
    covered = clip_samples + (n - 1) * hop_samples
    if pad and covered < total_samples:
        n += 1
    return n
