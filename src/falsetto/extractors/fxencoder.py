"""FXencoder feature extractor (TASKS.md T-15).

FXencoder (Koo et al., *Music Mixing Style Transfer*) is a **frozen**, stereo,
44.1 kHz encoder producing a **2048-dim** audio-effects embedding. Its pretrained
weights ship with the original GitHub repo, *not* HuggingFace.

Availability (verified-early policy): the real weights are not pip-installable, so
this module provides a **reference architecture** that runs end-to-end (correct
shapes, frozen, grads disabled) and a best-effort loader:

  * ``cfg.pretrained`` pointing at a local ``.pt``/``.ckpt`` -> weights are loaded
    (strict where possible, else non-strict with a warning);
  * otherwise the encoder is randomly initialized and :attr:`has_pretrained` is
    ``False`` — shapes are correct but features are not meaningful until real
    weights are supplied. FXencoder is therefore treated as *optional*; MERT is
    the primary extractor.

Output layout (T-15 decision): returns a **per-subsegment sequence** ``[N, 2048]``
by default so the FX-Segment transformer (T-19) has a short sequence to attend
over; set ``returns_sequence=False`` in config for a single ``[2048]`` per clip.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config.schema import ExtractorConfig
from ..utils.logging import get_logger
from .base import FeatureExtractor, register_extractor

_log = get_logger("extractors.fxencoder")


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_extractor("fxencoder")
class FXencoderExtractor(FeatureExtractor):
    """Reference FXencoder: stereo 44.1 kHz -> 2048-dim effects embedding(s)."""

    def __init__(self, cfg: ExtractorConfig, num_subsegments: int = 16) -> None:
        super().__init__()
        self.sample_rate = cfg.sample_rate or 44100
        self.embed_dim = 2048
        self.returns_sequence = cfg.returns_sequence
        self.num_subsegments = num_subsegments
        self.has_pretrained = False

        channels = [2, 64, 128, 256, 512, 1024, 2048]
        self.frontend = nn.Sequential(
            *[_ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.AdaptiveAvgPool1d(num_subsegments if self.returns_sequence else 1)

        if cfg.pretrained and Path(cfg.pretrained).exists():
            self.load_checkpoint(cfg.pretrained)
        elif cfg.pretrained:
            _log.warning(
                "FXencoder checkpoint %r not found; using random init "
                "(features not meaningful — supply real Koo et al. weights).",
                cfg.pretrained,
            )

    def load_checkpoint(self, path: str | Path) -> None:
        """Best-effort load of pretrained FXencoder weights."""
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        self.has_pretrained = True
        if missing or unexpected:
            _log.warning(
                "FXencoder loaded non-strict: %d missing, %d unexpected keys",
                len(missing),
                len(unexpected),
            )
        else:
            _log.info("FXencoder weights loaded from %s", path)

    def _to_stereo(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.size(0) > 2:
            waveform = waveform[:2]
        return waveform

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._to_stereo(waveform).to(next(self.parameters()).device)
        feats = self.frontend(x.unsqueeze(0))  # [1, 2048, T']
        pooled = self.pool(feats).squeeze(0)  # [2048, N] or [2048, 1]
        seq = pooled.transpose(0, 1).contiguous()  # [N, 2048]
        return seq if self.returns_sequence else seq.mean(dim=0)
