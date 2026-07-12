"""Combined results tables (TASKS.md T-36).

Render one markdown report with a section per stage — Stage-1, Stage-2
(Segment Transformer / Paper 1), Stage-2 (Fusion / Paper 2) — each a table of the
six metrics, with an optional **baselines** column comparing each row's AUC to a
reference value.
"""

from __future__ import annotations

from pathlib import Path

from ..training.metrics import MetricResults

_METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
_METRIC_HEADERS = ["Acc", "Prec", "Recall", "F1", "AUC", "Spec"]


def render_section(
    title: str,
    rows: dict[str, MetricResults],
    baselines: dict[str, float] | None = None,
) -> str:
    headers = ["Model", *_METRIC_HEADERS]
    if baselines is not None:
        headers += ["Baseline AUC", "ΔAUC"]
    lines = [f"#### {title}", "", "| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for label, res in rows.items():
        d = res.as_dict()
        cells = [label] + [f"{d[m]:.4f}" for m in _METRIC_ORDER]
        if baselines is not None:
            base = baselines.get(label)
            if base is None:
                cells += ["—", "—"]
            else:
                cells += [f"{base:.4f}", f"{d['auc'] - base:+.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def comparison_report(
    sections: dict[str, dict[str, MetricResults]],
    baselines: dict[str, dict[str, float]] | None = None,
    title: str = "FALSETTO — results",
) -> str:
    """Render all sections into one markdown document.

    Args:
        sections: ``{section_title: {model_label: MetricResults}}``.
        baselines: optional ``{section_title: {model_label: baseline_auc}}``.
    """
    out = [f"# {title}", ""]
    for section_title, rows in sections.items():
        section_baselines = (baselines or {}).get(section_title)
        out.append(render_section(section_title, rows, section_baselines))
    return "\n".join(out)


def write_comparison(
    sections: dict[str, dict[str, MetricResults]],
    out_path: str | Path,
    baselines: dict[str, dict[str, float]] | None = None,
    title: str = "FALSETTO — results",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(comparison_report(sections, baselines, title))
    return out_path
