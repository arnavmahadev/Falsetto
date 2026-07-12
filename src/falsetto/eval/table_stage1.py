"""Stage-1 results table (TASKS.md T-23).

Evaluate one or more trained Stage-1 checkpoints on a manifest split and emit a
markdown table in the style of Paper 1 Table I (one row per extractor, the six
metrics as columns).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config.schema import Config
from ..training.metrics import MetricAccumulator, MetricResults
from ..utils.device import select_device
from ..utils.logging import get_logger

_log = get_logger("eval.table_stage1")

_METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
_METRIC_HEADERS = ["Acc", "Prec", "Recall", "F1", "AUC", "Spec"]


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> MetricResults:
    model.to(device).eval()
    acc = MetricAccumulator()
    for waveforms, labels in loader:
        logits = model(waveforms.to(device))
        acc.update(logits.float(), labels)
    return acc.compute()


def results_to_markdown(rows: dict[str, MetricResults], title: str = "Stage-1 results") -> str:
    """Render ``{label: MetricResults}`` as a markdown table."""
    header = "| Extractor | " + " | ".join(_METRIC_HEADERS) + " |"
    sep = "|" + "---|" * (len(_METRIC_HEADERS) + 1)
    lines = [f"### {title}", "", header, sep]
    for label, res in rows.items():
        d = res.as_dict()
        cells = " | ".join(f"{d[m]:.4f}" for m in _METRIC_ORDER)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"


def evaluate_checkpoint(cfg: Config, ckpt_path: str | Path, split: str = "test") -> MetricResults:
    """Load a checkpoint per ``cfg`` and evaluate it on ``split``."""
    from ..data.datasets import Stage1ClipDataset, make_dataloader
    from ..data.manifests import load_manifest
    from ..models.stage1 import build_stage1_model

    device = select_device(cfg.device)
    manifest = load_manifest(cfg.data.manifest)
    ds = Stage1ClipDataset(manifest, cfg.extractor.name, cfg.data.clip_seconds,
                           split=split, random_crop=False)
    loader = make_dataloader(ds, cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = build_stage1_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    return evaluate_model(model, loader, device)


def write_table(rows: dict[str, MetricResults], out_path: str | Path, title: str = "Stage-1 results") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = results_to_markdown(rows, title)
    out_path.write_text(md)
    _log.info("wrote table -> %s", out_path)
    return out_path
