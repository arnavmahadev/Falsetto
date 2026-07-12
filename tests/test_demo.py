"""Demo tests: synth generator, adaptive SSM, plots (network-free) + HF-gated analyzer."""

from __future__ import annotations

import os

import matplotlib
import pytest
import torch

from falsetto.demo.pipeline import DemoFeatures, DemoResult, adaptive_ssm
from falsetto.demo.plots import gate_figure, ssm_figure, waveform_figure
from falsetto.demo.synth import SR, generate_clip


def test_synth_shapes_and_determinism():
    a = generate_clip(seed=3, coherent=True, seconds=6.0)
    b = generate_clip(seed=3, coherent=True, seconds=6.0)
    assert a.shape == (1, int(6.0 * SR))
    assert torch.equal(a, b)  # deterministic per seed
    assert a.abs().max() <= 1.0 + 1e-5
    # coherent and incoherent differ
    c = generate_clip(seed=3, coherent=False, seconds=6.0)
    assert not torch.equal(a, c)


def test_adaptive_ssm_properties():
    E = torch.randn(12, 64)
    ssm = adaptive_ssm(E)
    assert ssm.shape == (12, 12)
    assert torch.allclose(ssm, ssm.T, atol=1e-5)
    assert torch.allclose(torch.diag(ssm), torch.ones(12), atol=1e-5)
    assert (ssm >= 0).all() and (ssm <= 1 + 1e-5).all()


def _fake_features(n=10, seq=48, dim=32):
    E = torch.randn(seq, dim)
    mask = torch.ones(seq, dtype=torch.bool)
    mask[:n] = False
    return DemoFeatures(
        embeddings=E, mask=mask, ssm=adaptive_ssm(E, mask), n_segments=n,
        boundaries_sec=[(i * 1.5, (i + 1) * 1.5) for i in range(n)],
        downbeats=[0.5 * i for i in range(20)], segmentation="1.5s windows", duration_sec=15.0,
    )


def test_plots_render():
    feats = _fake_features()
    result = DemoResult(p_incoherent=0.8, coherence=0.2, band="Weak / drifting structure",
                        gate_per_segment=[0.5] * feats.n_segments, mean_gate=0.5, features=feats)
    for fig in (ssm_figure(feats), waveform_figure(torch.randn(1, SR * 15), SR, feats), gate_figure(result)):
        assert fig is not None
        matplotlib.pyplot.close(fig)


@pytest.mark.skipif(os.environ.get("FALSETTO_RUN_HF") != "1", reason="needs MERT download")
def test_demo_analyzer_end_to_end():
    from falsetto.config.schema import ExtractorConfig
    from falsetto.demo.pipeline import DemoAnalyzer
    from falsetto.extractors import build_extractor
    from falsetto.models.fusion import FusionSegmentTransformer

    mert = build_extractor(ExtractorConfig(name="mert"))
    model = FusionSegmentTransformer(embed_dim=768, seq_len=48, d_model=64, n_heads=8,
                                     emb_stream_layers=1, ssm_stream_layers=1)
    analyzer = DemoAnalyzer(mert, model, torch.device("cpu"))
    result = analyzer.analyze(generate_clip(seed=0, coherent=True, seconds=8.0), SR)
    assert 0.0 <= result.p_incoherent <= 1.0
    assert result.features.n_segments > 0
    assert len(result.gate_per_segment) == result.features.n_segments
