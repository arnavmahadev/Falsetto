"""Stage-2 training (TASKS.md T-29 Segment Transformer, T-35 Fusion).

Trains a track-level classifier over segment-embedding sequences ``E`` (+ padding
mask); the model builds its own SSM internally. Shared by both Paper-1
(SegmentTransformer) and Paper-2 (FusionSegmentTransformer) via a small factory.
BCE, Fused Adam, early stopping, six metrics — same machinery as Stage-1.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config.schema import Config, ModelConfig, TrainConfig
from ..models.fusion import FusionSegmentTransformer
from ..models.segment_transformer import SegmentTransformer
from ..utils.device import amp_dtype, autocast
from ..utils.logging import ExperimentTracker, get_logger
from .losses import build_loss
from .metrics import MetricAccumulator, MetricResults
from .train_stage1 import build_optimizer

_log = get_logger("train.stage2")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Stage2SequenceDataset(Dataset):
    """Yields ``(E [N, D], mask [N], label)`` from precomputed segment sequences."""

    def __init__(self, items: list[dict]) -> None:
        # each item: {"embeddings": [N,D], "mask": [N] bool, "label": float}
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        label = torch.as_tensor(float(it["label"]), dtype=torch.float32)
        return it["embeddings"].float(), it["mask"].bool(), label

    @classmethod
    def from_cache_dir(cls, directory: str | Path) -> "Stage2SequenceDataset":
        directory = Path(directory)
        items = [torch.load(p, map_location="cpu") for p in sorted(directory.glob("*.pt"))]
        return cls(items)


def collate_sequences(batch):
    embs, masks, labels = zip(*batch, strict=True)
    return torch.stack(embs), torch.stack(masks), torch.stack(labels)


def make_seq_dataloader(ds: Stage2SequenceDataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_sequences)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_stage2_model(model_cfg: ModelConfig, embed_dim: int, seq_len: int = 48) -> nn.Module:
    name = model_cfg.name.lower()
    if name in ("segment_transformer", "segmenttransformer", "segment"):
        return SegmentTransformer(
            embed_dim=embed_dim, seq_len=seq_len, d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads, n_layers=model_cfg.n_layers, ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout, num_classes=model_cfg.num_classes,
        )
    if name in ("fusion_segment_transformer", "fusion", "fst"):
        return FusionSegmentTransformer(
            embed_dim=embed_dim, seq_len=seq_len, d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads, ffn_dim=model_cfg.ffn_dim,
            emb_stream_layers=model_cfg.emb_stream_layers,
            ssm_stream_layers=model_cfg.ssm_stream_layers,
            dropout=model_cfg.dropout, num_classes=model_cfg.num_classes,
            fusion=model_cfg.fusion,
        )
    raise ValueError(f"unknown Stage-2 model {model_cfg.name!r}")


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Stage2Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: torch.device,
        tracker: ExperimentTracker | None = None,
        run_name: str = "stage2",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.tracker = tracker
        self.run_name = run_name
        self.optimizer = build_optimizer([p for p in model.parameters() if p.requires_grad], cfg)
        self.loss_fn = build_loss(cfg.loss, focal_gamma=cfg.focal_gamma, focal_alpha=cfg.focal_alpha).to(device)
        self.use_amp = cfg.amp and device.type in ("cuda", "cpu", "mps")
        self._amp_dtype = amp_dtype(device)
        self._best = None
        self._best_epoch = -1
        self._patience_left = cfg.patience
        self.ckpt_path = Path(cfg.ckpt_dir) / run_name / "best.pt"

    def _is_better(self, value: float) -> bool:
        if self._best is None:
            return True
        if value != value:
            return False
        return value > self._best if self.cfg.monitor_mode == "max" else value < self._best

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for embs, mask, labels in loader:
            embs, mask, labels = embs.to(self.device), mask.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device, enabled=self.use_amp, dtype=self._amp_dtype):
                logits = self.model(embs, key_padding_mask=mask)
                loss = self.loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            total += loss.item() * labels.size(0)
            n += labels.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, MetricResults]:
        self.model.eval()
        acc = MetricAccumulator()
        total, n = 0.0, 0
        for embs, mask, labels in loader:
            embs, mask, labels = embs.to(self.device), mask.to(self.device), labels.to(self.device)
            logits = self.model(embs, key_padding_mask=mask)
            loss = self.loss_fn(logits, labels)
            total += loss.item() * labels.size(0)
            n += labels.size(0)
            acc.update(logits.float(), labels)
        return total / max(n, 1), acc.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        _log.info("training %s: %d epochs, monitor=%s(%s)", self.run_name, self.cfg.epochs,
                  self.cfg.monitor, self.cfg.monitor_mode)
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.evaluate(val_loader)
            monitored = metrics.as_dict().get(self.cfg.monitor, float("nan"))
            if self.tracker:
                self.tracker.log_scalar("train/loss", train_loss, epoch)
                self.tracker.log_scalar("val/loss", val_loss, epoch)
                self.tracker.log_scalars(metrics.as_dict(), epoch, prefix="val/")
            _log.info("epoch %d/%d | train=%.4f val=%.4f | acc=%.4f f1=%.4f auc=%.4f | %.1fs",
                      epoch + 1, self.cfg.epochs, train_loss, val_loss,
                      metrics.accuracy, metrics.f1, metrics.auc, time.time() - t0)
            if self._is_better(monitored):
                self._best, self._best_epoch, self._patience_left = monitored, epoch, self.cfg.patience
                self.save_checkpoint(epoch, metrics)
            else:
                self._patience_left -= 1
                if self.cfg.early_stopping and self._patience_left <= 0:
                    _log.info("early stopping at epoch %d (best %s=%.4f)", epoch + 1,
                              self.cfg.monitor, self._best)
                    break
        return self.ckpt_path

    def save_checkpoint(self, epoch: int, metrics: MetricResults) -> Path:
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "epoch": epoch,
                    "metrics": metrics.as_dict(), "train_config": asdict(self.cfg),
                    "monitor": self.cfg.monitor, "monitor_value": self._best}, self.ckpt_path)
        return self.ckpt_path


def train_stage2_from_sequences(
    cfg: Config,
    train_items: list[dict],
    val_items: list[dict],
    embed_dim: int,
    seq_len: int = 48,
) -> Path:
    """Train a Stage-2 model on precomputed segment sequences. Returns best ckpt path."""
    from ..utils.device import select_device
    from ..utils.seed import seed_everything

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = select_device(cfg.device)
    model = build_stage2_model(cfg.model, embed_dim, seq_len)
    train_loader = make_seq_dataloader(Stage2SequenceDataset(train_items), cfg.data.batch_size, True)
    val_loader = make_seq_dataloader(Stage2SequenceDataset(val_items), cfg.data.batch_size, False)
    tracker = None if cfg.tracker == "none" else ExperimentTracker(cfg.tracker, cfg.output_dir, cfg.name)
    trainer = Stage2Trainer(model, cfg.train, device, tracker, run_name=cfg.name)
    ckpt = trainer.fit(train_loader, val_loader)
    if tracker:
        tracker.close()
    return ckpt
