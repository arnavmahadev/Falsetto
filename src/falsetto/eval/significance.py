"""Significance testing (TASKS.md T-40).

Paired test comparing the Fusion Segment Transformer to the Segment Transformer
across test tracks (the paper reports a p-value ~0.09). Given per-track paired
values (e.g. per-track loss, squared error, or correctness) from the two models,
run a paired Wilcoxon signed-rank test (default) or a paired t-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignificanceResult:
    test: str
    statistic: float
    p_value: float
    n: int
    mean_diff: float

    def __str__(self) -> str:
        return (
            f"{self.test}: p={self.p_value:.4f} "
            f"(stat={self.statistic:.4f}, n={self.n}, mean_diff={self.mean_diff:+.4f})"
        )


def paired_significance(
    values_a: "np.ndarray | list[float]",
    values_b: "np.ndarray | list[float]",
    test: str = "wilcoxon",
) -> SignificanceResult:
    """Paired significance test between two models' per-track values.

    Args:
        values_a: per-track values for model A (e.g. FST).
        values_b: per-track values for model B (e.g. Segment Transformer), same order.
        test: ``"wilcoxon"`` (signed-rank, non-parametric) or ``"ttest"`` (paired t-test).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match shape: {a.shape} vs {b.shape}")
    from scipy import stats

    if test == "wilcoxon":
        diff = a - b
        if np.allclose(diff, 0):
            return SignificanceResult("wilcoxon", 0.0, 1.0, len(a), 0.0)
        res = stats.wilcoxon(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    elif test == "ttest":
        res = stats.ttest_rel(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    else:
        raise ValueError(f"unknown test {test!r} (expected 'wilcoxon' or 'ttest')")

    return SignificanceResult(test, stat, p, len(a), float(np.mean(a - b)))


def per_track_correct(probs: "np.ndarray | list[float]", labels: "np.ndarray | list[int]", threshold: float = 0.5) -> np.ndarray:
    """Per-track correctness (1.0/0.0) — a common paired value for the test."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (probs >= threshold).astype(int)
    return (preds == labels).astype(float)
