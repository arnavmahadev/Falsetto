"""Matplotlib figures for the demo: clean, academic styling.

Warm cream ground, a single slate-blue accent, slate/amber for the human/AI
semantic, thin de-emphasized spines, no gradients. Palette matches the results
site (docs/index.html). Generated headless so they can be produced and tested
without a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .pipeline import DemoFeatures, DemoResult  # noqa: E402

# Slate accent on warm cream, matching the results-site tokens (docs/index.html).
ACCENT = "#47567C"
TEAL = ACCENT  # backwards-compatible alias
INK = "#1B1915"
MUTED = "#6A6459"
GRID = "#E1DBCC"
PAPER = "#FCFAF4"  # the site's --surface, so plots blend into their gr.Plot blocks
HUMAN = "#47567C"  # --accent
AI = "#946517"     # --pending (amber)

# Cream -> slate -> deep sequential map.
SSM_CMAP = LinearSegmentedColormap.from_list("falsetto", ["#F4F1E9", "#A6B0CC", ACCENT, "#232B40"])

_BASE_RC = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": "#C9C1AE",
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _style(fig, ax):
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def ssm_figure(features: DemoFeatures):
    """Heatmap of the self-similarity matrix over real segments."""
    n = max(features.n_segments, 1)
    ssm = features.ssm[:n, :n].cpu().numpy()
    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(4.5, 4.1))
        im = ax.imshow(ssm, cmap=SSM_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Self-similarity matrix", fontsize=11, weight="bold", pad=8)
        ax.set_xlabel("segment j")
        ax.set_ylabel("segment i")
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("similarity  exp(−‖eᵢ−eⱼ‖²/s)", fontsize=9, color=MUTED)
        cbar.outline.set_edgecolor(GRID)
        fig.tight_layout()
    return fig


def waveform_figure(waveform: torch.Tensor, sr: int, features: DemoFeatures):
    """Waveform with detected downbeats and segment shading."""
    y = waveform.squeeze().cpu().numpy()
    if y.ndim > 1:
        y = y.mean(0)
    t = np.arange(len(y)) / sr
    step = max(1, len(y) // 4000)
    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.plot(t[::step], y[::step], color=TEAL, lw=0.6)
        for i, (s, e) in enumerate(features.boundaries_sec):
            if i % 2 == 0:
                ax.axvspan(s, e, color=TEAL, alpha=0.05)
        for db in features.downbeats:
            ax.axvline(db, color=MUTED, lw=0.4, alpha=0.45)
        ax.set_xlim(0, t[-1] if len(t) else 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_title(f"Waveform · {len(features.downbeats)} downbeats · "
                     f"{features.n_segments} segments", fontsize=10, weight="bold")
        ax.tick_params(length=0)
        _style(fig, ax)
    return fig


def gate_figure(result: DemoResult):
    """Per-segment fusion gate: content vs. structure weighting."""
    g = result.gate_per_segment or [0.5]
    x = np.arange(len(g))
    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.bar(x, g, color=TEAL, alpha=0.85, width=0.85)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        ax.set_ylim(0, 1)
        ax.set_xlabel("segment index")
        ax.set_ylabel("gate G")
        ax.set_title("Fusion gate per segment  (1 = content · 0 = structure)",
                     fontsize=10, weight="bold")
        ax.tick_params(length=0)
        _style(fig, ax)
    return fig
