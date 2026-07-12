"""Per-extractor audio loading and resampling (TASKS.md T-07).

Each Stage-1 feature extractor expects audio at a specific sample rate and
channel layout:

    16 kHz mono    — Wav2Vec 2.0, Music2Vec, SONICS baseline
    24 kHz mono    — MERT, Muffin
    44.1 kHz stereo — FXencoder

:data:`EXTRACTOR_AUDIO_SPEC` is the single source of truth for that mapping, and
:func:`load_for_extractor` returns a waveform already at the right rate/channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..utils.audio_io import load_audio, peak_normalize, resample, to_mono


@dataclass(frozen=True)
class AudioSpec:
    """Target audio format for one extractor."""

    sample_rate: int
    stereo: bool


EXTRACTOR_AUDIO_SPEC: dict[str, AudioSpec] = {
    "wav2vec2": AudioSpec(16000, stereo=False),
    "music2vec": AudioSpec(16000, stereo=False),
    "sonics": AudioSpec(16000, stereo=False),
    "mert": AudioSpec(24000, stereo=False),
    "muffin": AudioSpec(24000, stereo=False),
    "fxencoder": AudioSpec(44100, stereo=True),
    "dummy": AudioSpec(24000, stereo=False),  # network-free test/dry-run extractor
}


def spec_for(extractor: str) -> AudioSpec:
    """Return the :class:`AudioSpec` for an extractor name (case-insensitive)."""
    try:
        return EXTRACTOR_AUDIO_SPEC[extractor.lower()]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(
            f"no audio spec for extractor {extractor!r}; "
            f"known: {sorted(EXTRACTOR_AUDIO_SPEC)}"
        ) from exc


def conform(
    waveform: torch.Tensor,
    orig_sr: int,
    spec: AudioSpec,
    normalize: bool = False,
) -> torch.Tensor:
    """Resample + channel-conform an already-loaded ``[channels, samples]`` waveform.

    Mono targets are downmixed; stereo targets duplicate a mono source to two
    channels. Resampling happens after channel conforming so we resample the
    minimum number of channels.
    """
    if spec.stereo:
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.size(0) > 2:
            waveform = waveform[:2]
    else:
        waveform = to_mono(waveform)

    if orig_sr != spec.sample_rate:
        waveform = resample(waveform, orig_sr, spec.sample_rate)
    if normalize:
        waveform = peak_normalize(waveform)
    return waveform.contiguous()


def load_for_extractor(
    path: str | Path,
    extractor: str,
    normalize: bool = False,
) -> tuple[torch.Tensor, int]:
    """Load an audio file conformed to ``extractor``'s rate and channel layout.

    Returns ``(waveform, sample_rate)`` where the sample rate is the extractor's
    target rate.
    """
    spec = spec_for(extractor)
    waveform, file_sr = load_audio(path, sr=None, mono=False)
    waveform = conform(waveform, file_sr, spec, normalize=normalize)
    return waveform, spec.sample_rate
