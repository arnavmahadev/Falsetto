"""Training: losses, metrics, Stage-1 loop."""

from .losses import BCEWithLogits, FocalLoss, build_loss
from .metrics import MetricAccumulator, MetricResults, compute_metrics
from .train_stage1 import Stage1Trainer, build_optimizer, train_stage1_from_config
from .train_stage2 import (
    Stage2SequenceDataset,
    Stage2Trainer,
    build_stage2_model,
    train_stage2_from_sequences,
)

__all__ = [
    "BCEWithLogits",
    "FocalLoss",
    "build_loss",
    "compute_metrics",
    "MetricResults",
    "MetricAccumulator",
    "Stage1Trainer",
    "build_optimizer",
    "train_stage1_from_config",
    "Stage2Trainer",
    "Stage2SequenceDataset",
    "build_stage2_model",
    "train_stage2_from_sequences",
]
