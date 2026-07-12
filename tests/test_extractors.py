"""Phase 2 extractor tests (network-free parts) + opt-in HF smoke test."""

from __future__ import annotations

import math
import os

import pytest
import torch

from falsetto.config.schema import ExtractorConfig
from falsetto.extractors import available_extractors, build_extractor
from falsetto.extractors.base import FeatureExtractor


def _tone(sr: int, seconds: float, freq: float = 220.0, channels: int = 1) -> torch.Tensor:
    t = torch.arange(int(sr * seconds)) / sr
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0).repeat(channels, 1)


def test_registry_has_all_extractors():
    names = available_extractors()
    for expected in ["mert", "wav2vec2", "music2vec", "fxencoder", "muffin", "dummy"]:
        assert expected in names


def test_dummy_extractor_shapes():
    ext = build_extractor(ExtractorConfig(name="dummy", embed_dim=768, sample_rate=24000))
    feats = ext.extract(_tone(24000, 2.0))
    assert feats.dim() == 2 and feats.shape[1] == 768
    assert ext.pooled(_tone(24000, 1.0)).shape == (768,)


def test_fxencoder_frozen_2048_sequence():
    cfg = ExtractorConfig(name="fxencoder", sample_rate=44100, freeze=True)
    ext = build_extractor(cfg)  # build_extractor freezes when cfg.freeze
    assert isinstance(ext, FeatureExtractor)
    # grads disabled (frozen)
    assert all(not p.requires_grad for p in ext.parameters())
    assert ext.has_pretrained is False  # no real Koo et al. weights supplied
    seq = ext.extract(_tone(44100, 2.0, channels=2))
    assert seq.shape == (16, 2048)  # per-subsegment sequence


def test_fxencoder_single_vector_mode():
    cfg = ExtractorConfig(name="fxencoder", returns_sequence=False)
    ext = build_extractor(cfg)
    vec = ext.extract(_tone(44100, 2.0, channels=1))  # mono upmixed to stereo
    assert vec.shape == (2048,)


def test_muffin_three_band_sequence():
    cfg = ExtractorConfig(name="muffin", sample_rate=24000, embed_dim=512)
    ext = build_extractor(cfg)
    feats = ext.extract(_tone(24000, 2.0))
    assert feats.dim() == 2 and feats.shape[1] == 512
    # band bins cover the full mel range without gaps
    assert ext._band_bins[0][0] == 0
    assert ext._band_bins[-1][1] == ext.n_mels


@pytest.mark.skipif(
    os.environ.get("FALSETTO_RUN_HF") != "1",
    reason="set FALSETTO_RUN_HF=1 to download + smoke-test HuggingFace extractors",
)
@pytest.mark.parametrize("name,pretrained,sr,dim", [
    ("wav2vec2", "facebook/wav2vec2-base", 16000, 768),
    ("mert", "m-a-p/MERT-v1-95M", 24000, 768),
])
def test_hf_extractor_smoke(name, pretrained, sr, dim):
    cfg = ExtractorConfig(name=name, pretrained=pretrained, sample_rate=sr, embed_dim=dim)
    ext = build_extractor(cfg)
    feats = ext.extract(_tone(sr, 10.0))
    assert feats.dim() == 2 and feats.shape[1] == ext.embed_dim
