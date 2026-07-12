"""Ablation drivers + delta reporting (TASKS.md T-37, T-38).

- T-37 segmentation: 4-bar downbeat vs fixed-window segment sequences.
- T-38 fusion: gated (GMU) vs plain (mean) cross-attention fusion.

Each driver trains two variants that differ only in the ablated component and
reports the metric delta. Delta reporting is separated out (:func:`report_delta`)
so it can be unit-tested without training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config.schema import Config
from ..training.metrics import MetricResults
from ..utils.logging import get_logger

_log = get_logger("eval.ablations")


@dataclass
class AblationDelta:
    name: str
    baseline_label: str
    variant_label: str
    metric: str
    baseline: float
    variant: float

    @property
    def delta(self) -> float:
        return self.variant - self.baseline

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.baseline_label}={self.baseline:.4f} -> "
            f"{self.variant_label}={self.variant:.4f}  (delta {self.delta:+.4f})"
        )


def report_delta(
    name: str,
    baseline: MetricResults,
    variant: MetricResults,
    baseline_label: str,
    variant_label: str,
    metric: str = "auc",
) -> AblationDelta:
    return AblationDelta(
        name=name,
        baseline_label=baseline_label,
        variant_label=variant_label,
        metric=metric,
        baseline=baseline.as_dict()[metric],
        variant=variant.as_dict()[metric],
    )


def _train_and_eval(cfg: Config, train_items, val_items, test_items, embed_dim, seq_len) -> MetricResults:
    from ..training.train_stage2 import build_stage2_model, make_seq_dataloader
    from ..training.train_stage2 import Stage2SequenceDataset, Stage2Trainer
    from ..eval.table_stage1 import MetricAccumulator
    from ..utils.device import select_device

    device = select_device(cfg.device)
    model = build_stage2_model(cfg.model, embed_dim, seq_len)
    trainer = Stage2Trainer(model, cfg.train, device, tracker=None, run_name=cfg.name)
    train_loader = make_seq_dataloader(Stage2SequenceDataset(train_items), cfg.data.batch_size, True)
    val_loader = make_seq_dataloader(Stage2SequenceDataset(val_items), cfg.data.batch_size, False)
    trainer.fit(train_loader, val_loader)
    # Evaluate best on test.
    state = torch.load(trainer.ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    test_loader = make_seq_dataloader(Stage2SequenceDataset(test_items), cfg.data.batch_size, False)
    model.eval()
    acc = MetricAccumulator()
    with torch.no_grad():
        for embs, mask, labels in test_loader:
            logits = model(embs.to(device), key_padding_mask=mask.to(device))
            acc.update(logits.float(), labels)
    return acc.compute()


def run_fusion_ablation(cfg: Config, train_items, val_items, test_items, embed_dim, seq_len=48, metric="auc") -> AblationDelta:
    """T-38: gated (GMU) vs plain (mean) fusion, all else equal."""
    gmu_cfg = copy.deepcopy(cfg)
    gmu_cfg.model.name = "fusion_segment_transformer"
    gmu_cfg.model.fusion = "gmu"
    gmu_cfg.name = f"{cfg.name}_gmu"
    gated = _train_and_eval(gmu_cfg, train_items, val_items, test_items, embed_dim, seq_len)

    plain_cfg = copy.deepcopy(cfg)
    plain_cfg.model.name = "fusion_segment_transformer"
    plain_cfg.model.fusion = "mean"
    plain_cfg.name = f"{cfg.name}_plain"
    plain = _train_and_eval(plain_cfg, train_items, val_items, test_items, embed_dim, seq_len)

    delta = report_delta("fusion (T-38)", plain, gated, "plain-cross-attn", "gated-GMU", metric)
    _log.info(str(delta))
    return delta


def write_ablation_report(deltas: list[AblationDelta], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["### Ablations", "", "| Ablation | Baseline | Variant | Metric | Baseline | Variant | Delta |",
             "|---|---|---|---|---|---|---|"]
    for d in deltas:
        lines.append(
            f"| {d.name} | {d.baseline_label} | {d.variant_label} | {d.metric} | "
            f"{d.baseline:.4f} | {d.variant:.4f} | {d.delta:+.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
