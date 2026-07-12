"""Fusion-gate visualization (TASKS.md T-39, Paper 2 Fig. 3).

Two plots from the GMU gate ``G`` (T-33):

  1. Histogram of each track's **mean gate weight**, Real vs. Fake.
  2. **Segment-wise** mean gate curve (gate weight vs. segment index), Real vs. Fake.

``G`` is ``[N, d]`` per track (or batched ``[B, N, d]``); we reduce the feature
axis to a scalar per segment. Uses a non-interactive matplotlib backend so it
runs headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def reduce_gate_per_segment(gate: torch.Tensor) -> np.ndarray:
    """``[N, d]`` (or ``[B, N, d]``) -> per-segment scalar by averaging the feature axis."""
    g = gate.detach().float().cpu()
    if g.dim() == 3:
        return g.mean(dim=-1).numpy()  # [B, N]
    return g.mean(dim=-1).numpy()  # [N]


def mean_gate_per_track(gate: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Single scalar: mean gate over (unmasked) segments of one track."""
    per_seg = reduce_gate_per_segment(gate)  # [N]
    if mask is not None:
        keep = (~mask.cpu().numpy()).astype(bool)
        per_seg = per_seg[keep] if keep.any() else per_seg
    return float(np.mean(per_seg))


def gate_histogram(
    real_means: list[float],
    fake_means: list[float],
    out_path: str | Path,
    bins: int = 30,
) -> Path:
    """Overlaid histogram of per-track mean gate weight, Real vs. Fake."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(real_means, bins=bins, alpha=0.6, label="Real", color="#2c7fb8", density=True)
    ax.hist(fake_means, bins=bins, alpha=0.6, label="Fake", color="#de2d26", density=True)
    ax.set_xlabel("Mean gate weight G (content vs. structure)")
    ax.set_ylabel("Density")
    ax.set_title("Fusion gate distribution: Real vs. Fake")
    ax.legend()
    return _save(fig, out_path)


def segmentwise_gate_curve(
    real_curves: np.ndarray,
    fake_curves: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Mean +/- std gate weight per segment index, Real vs. Fake.

    ``real_curves`` / ``fake_curves`` are ``[num_tracks, N]`` arrays of per-segment
    gate weights.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for curves, label, color in [
        (real_curves, "Real", "#2c7fb8"),
        (fake_curves, "Fake", "#de2d26"),
    ]:
        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Gate weight G")
    ax.set_title("Segment-wise fusion gate: Real vs. Fake")
    ax.legend()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
