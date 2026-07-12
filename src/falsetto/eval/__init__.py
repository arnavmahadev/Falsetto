"""Evaluation: results tables, ablations, gate viz, significance tests."""

from .ablations import AblationDelta, report_delta, run_fusion_ablation, write_ablation_report
from .compare import comparison_report, render_section, write_comparison
from .gate_viz import (
    gate_histogram,
    mean_gate_per_track,
    reduce_gate_per_segment,
    segmentwise_gate_curve,
)
from .significance import SignificanceResult, paired_significance, per_track_correct
from .table_stage1 import (
    evaluate_checkpoint,
    evaluate_model,
    results_to_markdown,
    write_table,
)

__all__ = [
    "evaluate_model",
    "evaluate_checkpoint",
    "results_to_markdown",
    "write_table",
    "comparison_report",
    "render_section",
    "write_comparison",
    "report_delta",
    "run_fusion_ablation",
    "write_ablation_report",
    "AblationDelta",
    "gate_histogram",
    "segmentwise_gate_curve",
    "reduce_gate_per_segment",
    "mean_gate_per_track",
    "paired_significance",
    "per_track_correct",
    "SignificanceResult",
]
