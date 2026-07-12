"""Phase 6 tests: comparison tables, gate viz, significance, ablation delta."""

from __future__ import annotations

import numpy as np
import torch

from falsetto.eval.ablations import report_delta, write_ablation_report
from falsetto.eval.compare import comparison_report
from falsetto.eval.gate_viz import (
    gate_histogram,
    mean_gate_per_track,
    reduce_gate_per_segment,
    segmentwise_gate_curve,
)
from falsetto.eval.significance import paired_significance, per_track_correct
from falsetto.training.metrics import MetricResults


# T-36 compare
def test_comparison_report_sections_and_baselines():
    sections = {
        "Stage-1 (FakeMusicCaps)": {"MERT": MetricResults(accuracy=0.98, auc=0.99)},
        "Stage-2 Fusion (AIME)": {"MERT-FST": MetricResults(accuracy=0.96, auc=0.987)},
    }
    baselines = {"Stage-2 Fusion (AIME)": {"MERT-FST": 0.985}}
    md = comparison_report(sections, baselines)
    assert "#### Stage-1 (FakeMusicCaps)" in md
    assert "#### Stage-2 Fusion (AIME)" in md
    assert "Baseline AUC" in md
    assert "+0.0020" in md  # 0.987 - 0.985 delta


# T-39 gate viz
def test_gate_reduction_and_plots(tmp_path):
    G = torch.rand(48, 64)  # [N, d]
    per_seg = reduce_gate_per_segment(G)
    assert per_seg.shape == (48,)
    assert 0.0 <= mean_gate_per_track(G) <= 1.0

    real_means = list(np.random.rand(20) * 0.4 + 0.5)
    fake_means = list(np.random.rand(20) * 0.4 + 0.1)
    hist = gate_histogram(real_means, fake_means, tmp_path / "hist.png")
    assert hist.exists()

    real_curves = np.random.rand(10, 48)
    fake_curves = np.random.rand(12, 48)
    curve = segmentwise_gate_curve(real_curves, fake_curves, tmp_path / "curve.png")
    assert curve.exists()


# T-40 significance
def test_paired_significance_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, size=50)  # model A consistently higher
    b = rng.normal(0.0, 0.1, size=50)
    res = paired_significance(a, b, test="wilcoxon")
    assert res.p_value < 0.05
    assert res.n == 50
    assert res.mean_diff > 0

    res_t = paired_significance(a, b, test="ttest")
    assert res_t.p_value < 0.05


def test_per_track_correct():
    probs = [0.9, 0.1, 0.8, 0.2]
    labels = [1, 0, 0, 1]
    correct = per_track_correct(probs, labels)
    assert list(correct) == [1.0, 1.0, 0.0, 0.0]


# T-37/T-38 ablation delta reporting
def test_report_delta_and_write(tmp_path):
    base = MetricResults(auc=0.9850)
    variant = MetricResults(auc=0.9867)
    d = report_delta("fusion (T-38)", base, variant, "plain", "gated", metric="auc")
    assert abs(d.delta - 0.0017) < 1e-6
    out = write_ablation_report([d], tmp_path / "abl.md")
    assert out.exists() and "fusion (T-38)" in out.read_text()
