"""Common feature-extractor interface (TASKS.md Phase 2).

Every Stage-1 encoder implements :class:`FeatureExtractor`:

    class FeatureExtractor:
        sample_rate: int
        embed_dim: int
        returns_sequence: bool
        def extract(waveform) -> Tensor   # [T, D] sequence, or [D] pooled

Extractors are ``nn.Module`` subclasses so they can be frozen and moved to a
device uniformly. A small registry + :func:`build_extractor` lets configs select
one by name. :class:`DummyExtractor` is a deterministic, network-free stand-in
for tests and shape checks.
"""

from __future__ import annotations

import abc
from typing import Callable

import torch
import torch.nn as nn

from ..config.schema import ExtractorConfig
from ..utils.audio_io import to_mono


class FeatureExtractor(nn.Module, abc.ABC):
    """Abstract Stage-1 feature extractor."""

    sample_rate: int
    embed_dim: int
    returns_sequence: bool

    @abc.abstractmethod
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map a single ``[channels, samples]`` (or ``[samples]``) clip to features.

        Returns ``[T, D]`` when :attr:`returns_sequence`, else ``[D]``.
        """

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.extract(waveform)

    def pooled(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mean-pooled ``[D]`` embedding regardless of :attr:`returns_sequence`."""
        feats = self.extract(waveform)
        return feats.mean(dim=0) if feats.dim() == 2 else feats

    def freeze(self) -> "FeatureExtractor":
        """Disable grads and switch to eval mode (frozen extractor)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    @staticmethod
    def _as_mono_1d(waveform: torch.Tensor) -> torch.Tensor:
        """Coerce ``[channels, samples]`` / ``[samples]`` to a 1-D mono tensor."""
        if waveform.dim() == 2:
            waveform = to_mono(waveform).squeeze(0)
        return waveform


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[[ExtractorConfig], FeatureExtractor]] = {}


def register_extractor(name: str) -> Callable[[type], type]:
    """Class decorator registering an extractor factory under ``name``."""

    def deco(cls: type) -> type:
        _REGISTRY[name.lower()] = cls  # type: ignore[assignment]
        return cls

    return deco


def build_extractor(cfg: ExtractorConfig) -> FeatureExtractor:
    """Instantiate the extractor named by ``cfg.name``."""
    key = cfg.name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown extractor {cfg.name!r}; registered: {sorted(_REGISTRY)}")
    extractor = _REGISTRY[key](cfg)  # type: ignore[call-arg]
    if cfg.freeze:
        extractor.freeze()
    return extractor


def available_extractors() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
# Dummy extractor (tests / dry-runs, no network)
# --------------------------------------------------------------------------- #
@register_extractor("dummy")
class DummyExtractor(FeatureExtractor):
    """Deterministic extractor: frames the waveform and projects to ``embed_dim``.

    Produces ``[T, D]`` where ``T`` scales with input length, so downstream shape
    logic can be tested without downloading a pretrained model.
    """

    def __init__(self, cfg: ExtractorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ExtractorConfig(name="dummy")
        self.sample_rate = cfg.sample_rate or 24000
        self.embed_dim = cfg.embed_dim or 768
        self.returns_sequence = True
        self.frame = 320  # ~wav2vec2-like stride
        self.proj = nn.Linear(self.frame, self.embed_dim)

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._as_mono_1d(waveform)
        n_frames = max(1, x.numel() // self.frame)
        x = x[: n_frames * self.frame].reshape(n_frames, self.frame)
        return self.proj(x)
