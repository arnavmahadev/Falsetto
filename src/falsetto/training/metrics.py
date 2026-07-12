"""Classification metrics from logits + labels (TASKS.md T-21).

Computes the six metrics reported throughout the papers — Accuracy, Precision,
Recall, F1, AUC, Specificity — matching scikit-learn on well-defined inputs.
AUC uses sklearn's ``roc_auc_score`` (returns NaN if only one class is present).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MetricResults:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = float("nan")
    specificity: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "specificity": self.specificity,
        }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=float)


def compute_metrics(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> MetricResults:
    """Compute all six metrics.

    Args:
        logits: raw logits (sigmoid applied here), shape ``[N]`` or ``[N, 1]``.
        targets: binary labels (0=real, 1=ai), shape ``[N]``.
        threshold: probability threshold for the positive (ai) class.
    """
    logits = _to_numpy(logits).reshape(-1)
    targets = _to_numpy(targets).reshape(-1).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    accuracy = (tp + tn) / max(len(targets), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auc = float("nan")
    if len(np.unique(targets)) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(targets, probs))
        except Exception:  # pragma: no cover
            auc = float("nan")

    return MetricResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        specificity=specificity,
        extra={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


class MetricAccumulator:
    """Collect logits/targets across batches, then compute metrics once."""

    def __init__(self) -> None:
        self._logits: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, logits, targets) -> None:
        self._logits.append(_to_numpy(logits).reshape(-1))
        self._targets.append(_to_numpy(targets).reshape(-1))

    def compute(self, threshold: float = 0.5) -> MetricResults:
        if not self._logits:
            return MetricResults()
        return compute_metrics(
            np.concatenate(self._logits), np.concatenate(self._targets), threshold
        )

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()
