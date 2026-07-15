"""Stage-1 training loop (TASKS.md T-22).

Trains a :class:`~falsetto.models.stage1.Stage1Detector` head (the extractor is
frozen): Adam (lr 1e-5, wd 1e-6 by default), AMP autocast, gradient clipping,
early stopping, and best-by-monitor-metric checkpointing. All six metrics are
logged to the experiment tracker each epoch.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.schema import Config, TrainConfig
from ..utils.device import amp_dtype, autocast
from ..utils.logging import ExperimentTracker, get_logger
from .losses import build_loss
from .metrics import MetricAccumulator, MetricResults

_log = get_logger("train.stage1")


def build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if name == "fused_adam":
        try:
            return torch.optim.Adam(params, fused=True, **kwargs)
        except (RuntimeError, ValueError):
            return torch.optim.Adam(params, **kwargs)
    return torch.optim.Adam(params, **kwargs)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: MetricResults


class Stage1Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: torch.device,
        tracker: ExperimentTracker | None = None,
        run_name: str = "stage1",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.tracker = tracker
        self.run_name = run_name

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("no trainable parameters (is the whole model frozen?)")
        self.optimizer = build_optimizer(trainable, cfg)
        self.loss_fn = build_loss(
            cfg.loss, focal_gamma=cfg.focal_gamma, focal_alpha=cfg.focal_alpha
        ).to(device)

        self.use_amp = cfg.amp and device.type in ("cuda", "cpu", "mps")
        self._amp_dtype = amp_dtype(device)
        # GradScaler only helps float16 on CUDA; bf16 / MPS / CPU run without it.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and device.type == "cuda" and self._amp_dtype == torch.float16,
        )
        self._best = None
        self._best_epoch = -1
        self._patience_left = cfg.patience
        self.ckpt_path = Path(cfg.ckpt_dir) / run_name / "best.pt"

    # ------------------------------------------------------------------ #
    def _is_better(self, value: float) -> bool:
        if self._best is None:
            return True
        if value != value:  # NaN
            return False
        return value > self._best if self.cfg.monitor_mode == "max" else value < self._best

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for step, (waveforms, labels) in enumerate(loader):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device, enabled=self.use_amp, dtype=self._amp_dtype):
                logits = self.model(waveforms)
                loss = self.loss_fn(logits, labels)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            total += loss.item() * labels.size(0)
            n += labels.size(0)
            if self.tracker and step % self.cfg.log_every == 0:
                self.tracker.log_scalar("train/loss_step", loss.item(), epoch * len(loader) + step)
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, MetricResults]:
        self.model.eval()
        acc = MetricAccumulator()
        total, n = 0.0, 0
        for waveforms, labels in loader:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            with autocast(self.device, enabled=self.use_amp, dtype=self._amp_dtype):
                logits = self.model(waveforms)
                loss = self.loss_fn(logits, labels)
            total += loss.item() * labels.size(0)
            n += labels.size(0)
            acc.update(logits.float(), labels)
        return total / max(n, 1), acc.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        _log.info(
            "training %s: %d epochs, monitor=%s(%s), device=%s",
            self.run_name, self.cfg.epochs, self.cfg.monitor, self.cfg.monitor_mode, self.device,
        )
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.evaluate(val_loader)
            monitored = metrics.as_dict().get(self.cfg.monitor, float("nan"))

            if self.tracker:
                self.tracker.log_scalar("train/loss", train_loss, epoch)
                self.tracker.log_scalar("val/loss", val_loss, epoch)
                self.tracker.log_scalars(metrics.as_dict(), epoch, prefix="val/")

            _log.info(
                "epoch %d/%d | train_loss=%.4f val_loss=%.4f | acc=%.4f f1=%.4f auc=%.4f spec=%.4f | %.1fs",
                epoch + 1, self.cfg.epochs, train_loss, val_loss,
                metrics.accuracy, metrics.f1, metrics.auc, metrics.specificity, time.time() - t0,
            )

            if self._is_better(monitored):
                self._best = monitored
                self._best_epoch = epoch
                self._patience_left = self.cfg.patience
                self.save_checkpoint(epoch, metrics)
            else:
                self._patience_left -= 1
                if self.cfg.early_stopping and self._patience_left <= 0:
                    _log.info("early stopping at epoch %d (best %s=%.4f @ epoch %d)",
                              epoch + 1, self.cfg.monitor, self._best, self._best_epoch + 1)
                    break
        return self.ckpt_path

    def save_checkpoint(self, epoch: int, metrics: MetricResults) -> Path:
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics.as_dict(),
                "train_config": asdict(self.cfg),
                "monitor": self.cfg.monitor,
                "monitor_value": self._best,
            },
            self.ckpt_path,
        )
        return self.ckpt_path


# --------------------------------------------------------------------------- #
# Config-driven entry point
# --------------------------------------------------------------------------- #
def train_stage1_from_config(cfg: Config, manifest=None, resume: str | Path | None = None) -> Path:
    """Build data + model from a :class:`Config` and train. Returns best ckpt path.

    ``resume`` reloads the weights from a previous checkpoint and carries its best
    monitor value forward, so a run cut short by a dropped session (a closed laptop,
    a reclaimed Colab VM) continues instead of paying for those epochs twice. The
    optimizer state is not in the checkpoint, so Adam's moments restart; that costs
    roughly an epoch of re-warming, which is far cheaper than starting over.
    """
    from ..data.datasets import Stage1ClipDataset, make_dataloader
    from ..data.manifests import load_manifest
    from ..data.augment import Augmentor
    from ..models.stage1 import build_stage1_model
    from ..utils.device import select_device
    from ..utils.seed import seed_everything

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = select_device(cfg.device)
    if manifest is None:
        manifest = load_manifest(cfg.data.manifest)

    aug = Augmentor.from_config(cfg.data, enabled=True, seed=cfg.seed)
    train_ds = Stage1ClipDataset(
        manifest, cfg.extractor.name, cfg.data.clip_seconds, split="train",
        augmentor=aug, random_crop=True, seed=cfg.seed,
    )
    val_ds = Stage1ClipDataset(
        manifest, cfg.extractor.name, cfg.data.clip_seconds, split="val", random_crop=False,
    )
    train_loader = make_dataloader(train_ds, cfg.data.batch_size, shuffle=True,
                                   num_workers=cfg.data.num_workers)
    val_loader = make_dataloader(val_ds, cfg.data.batch_size, shuffle=False,
                                 num_workers=cfg.data.num_workers)

    model = build_stage1_model(cfg)
    tracker = None
    if cfg.tracker != "none":
        tracker = ExperimentTracker(cfg.tracker, cfg.output_dir, cfg.name)
    trainer = Stage1Trainer(model, cfg.train, device, tracker, run_name=cfg.name)

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        # Carry the best score forward, or the first epoch would look like an
        # improvement on -inf and overwrite a better checkpoint with a worse one.
        prev = state.get("monitor_value")
        if prev is not None and state.get("monitor") == cfg.train.monitor:
            trainer._best = prev
        _log.info(
            "resumed from %s (epoch %s, %s=%.4f)",
            resume, state.get("epoch", "?") + 1 if isinstance(state.get("epoch"), int) else "?",
            state.get("monitor", "?"), prev if prev is not None else float("nan"),
        )

    ckpt = trainer.fit(train_loader, val_loader)
    if tracker:
        tracker.close()
    return ckpt
