"""Waveform augmentation for SSL extractors (TASKS.md T-09).

Two augmentations, each applied independently with probability ``p`` (0.5):

    - pitch shift, uniform in ``[-pitch_semitones, +pitch_semitones]`` (default +/-2)
    - time stretch, uniform in ``[time_stretch_min, time_stretch_max]`` (default 0.8-1.25x)

Both go through librosa's phase vocoder (pitch-preserving stretch). After a
time stretch the sample count changes, so the result is cropped/padded back to
the original length.

Gotchas enforced by design:
  * ``enabled`` gates everything — call sites pass ``training and extractor_is_ssl``.
    **Never augment FXencoder inputs** (frozen; augmentation distorts its features).
  * A seeded :class:`numpy.random.Generator` makes each draw reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config.schema import DataConfig


class Augmentor:
    """Randomized pitch-shift + time-stretch for SSL extractor inputs."""

    def __init__(
        self,
        aug_prob: float = 0.5,
        pitch_semitones: float = 2.0,
        time_stretch_min: float = 0.8,
        time_stretch_max: float = 1.25,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.aug_prob = aug_prob
        self.pitch_semitones = pitch_semitones
        self.time_stretch_min = time_stretch_min
        self.time_stretch_max = time_stretch_max
        self.enabled = enabled
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, cfg: DataConfig, enabled: bool = True, seed: int | None = None) -> "Augmentor":
        return cls(
            aug_prob=cfg.aug_prob,
            pitch_semitones=cfg.pitch_shift_semitones,
            time_stretch_min=cfg.time_stretch_min,
            time_stretch_max=cfg.time_stretch_max,
            enabled=cfg.augment and enabled,
            seed=seed,
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return an augmented copy of ``[channels, samples]`` (length preserved)."""
        if not self.enabled:
            return waveform
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)

        out = waveform
        if self._rng.random() < self.aug_prob:
            semitones = float(self._rng.uniform(-self.pitch_semitones, self.pitch_semitones))
            out = self._pitch_shift(out, sample_rate, semitones)
        if self._rng.random() < self.aug_prob:
            rate = float(self._rng.uniform(self.time_stretch_min, self.time_stretch_max))
            out = self._time_stretch(out, rate, target_len=waveform.size(-1))

        return out.squeeze(0) if squeeze else out

    def _pitch_shift(self, waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
        import librosa

        arr = waveform.cpu().numpy()
        shifted = np.stack(
            [librosa.effects.pitch_shift(ch, sr=sr, n_steps=semitones) for ch in arr],
            axis=0,
        )
        return torch.from_numpy(shifted).to(waveform.dtype)

    def _time_stretch(self, waveform: torch.Tensor, rate: float, target_len: int) -> torch.Tensor:
        import librosa

        arr = waveform.cpu().numpy()
        stretched = np.stack(
            [librosa.effects.time_stretch(ch, rate=rate) for ch in arr],
            axis=0,
        )
        out = torch.from_numpy(stretched).to(waveform.dtype)
        return _fit_length(out, target_len)


def _fit_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Crop or zero-pad ``[channels, samples]`` to exactly ``length`` samples."""
    samples = waveform.size(-1)
    if samples == length:
        return waveform
    if samples > length:
        return waveform[..., :length]
    pad = waveform.new_zeros(waveform.size(0), length - samples)
    return torch.cat([waveform, pad], dim=-1)
