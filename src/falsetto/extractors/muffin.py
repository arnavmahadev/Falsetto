"""Muffin encoder feature extractor (TASKS.md T-16 — optional, hardest).

Muffin (Ng et al., 2025) operates on a mel-spectrogram of **0-12 kHz @ 24 kHz**
split into three frequency bands — low **0-2 kHz**, mid **2-6 kHz**, high
**6-12 kHz** — each encoded and fused. The full recipe pre-trains the encoder
(MLP head + FFT band filtering; lr 1e-5, wd 1e-6), fine-tunes (lr 2e-2, wd 5e-2),
then freezes it as an AudioCAT encoder.

Scope note: reference Muffin weights are not confirmed public, so the pretrain /
fine-tune loops are **deferred** (a from-scratch pretrain is a large budget item).
This module implements the concrete, testable part — the 3-band mel front-end and
a small fusion encoder — producing a frozen ``[T, D]`` sequence usable by
AudioCAT. :attr:`has_pretrained` is ``False`` until real weights are trained/loaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import ExtractorConfig
from .base import FeatureExtractor, register_extractor

# Band edges in Hz over the 0-12 kHz analysis range.
BANDS_HZ = ((0, 2000), (2000, 6000), (6000, 12000))


class _BandEncoder(nn.Module):
    def __init__(self, in_mels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_mels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, mels, T] -> [B, out, T]
        return self.net(x)


@register_extractor("muffin")
class MuffinExtractor(FeatureExtractor):
    """3-band mel encoder (reference implementation; pretrain deferred)."""

    def __init__(self, cfg: ExtractorConfig, n_mels: int = 128, band_dim: int = 256) -> None:
        super().__init__()
        import torchaudio

        self.sample_rate = cfg.sample_rate or 24000
        self.embed_dim = cfg.embed_dim or 768
        self.returns_sequence = True
        self.has_pretrained = False
        self.n_mels = n_mels

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            f_min=0.0,
            f_max=12000.0,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        # Map each band's Hz range to mel-bin index ranges (linear over mel index).
        self._band_bins = self._band_bin_ranges(n_mels)
        self.band_encoders = nn.ModuleList(
            [_BandEncoder(hi - lo, band_dim) for (lo, hi) in self._band_bins]
        )
        self.fuse = nn.Conv1d(band_dim * len(BANDS_HZ), self.embed_dim, kernel_size=1)

    def _band_bin_ranges(self, n_mels: int) -> list[tuple[int, int]]:
        f_max = 12000.0
        ranges = []
        for lo_hz, hi_hz in BANDS_HZ:
            lo = int(round(n_mels * lo_hz / f_max))
            hi = int(round(n_mels * hi_hz / f_max))
            ranges.append((lo, max(hi, lo + 1)))
        return ranges

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self._as_mono_1d(waveform).to(device)
        mel = self.to_db(self.melspec(x))  # [n_mels, T]
        mel = mel.unsqueeze(0)  # [1, n_mels, T]

        band_feats = []
        for (lo, hi), enc in zip(self._band_bins, self.band_encoders, strict=True):
            band_feats.append(enc(mel[:, lo:hi, :]))
        fused = self.fuse(torch.cat(band_feats, dim=1))  # [1, D, T]
        return fused.squeeze(0).transpose(0, 1).contiguous()  # [T, D]
