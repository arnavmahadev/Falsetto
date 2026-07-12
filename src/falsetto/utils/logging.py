"""Logging + experiment tracking.

Two things live here:

- :func:`get_logger` — a plain stdout logger with a consistent format.
- :class:`ExperimentTracker` — a thin wrapper over TensorBoard or Weights &
  Biases (selected by config) exposing ``log_scalar`` / ``log_scalars`` /
  ``close``. ``tracker="none"`` gives a silent no-op so training code never has
  to branch on whether tracking is enabled.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str = "falsetto", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root = logging.getLogger("falsetto")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("falsetto") else f"falsetto.{name}")


class ExperimentTracker:
    """Unified scalar tracker over TensorBoard / W&B / none."""

    def __init__(
        self,
        backend: str = "tensorboard",
        log_dir: str | Path = "runs",
        run_name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.backend = backend.lower()
        self.log_dir = Path(log_dir)
        self.run_name = run_name or "run"
        self._writer = None
        self._wandb = None
        self._log = get_logger("tracker")

        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.log_dir / self.run_name))
            if config:
                # Store config as text so it shows up in the run.
                self._writer.add_text("config", _as_markdown(config))
        elif self.backend == "wandb":
            import wandb  # optional dependency

            self._wandb = wandb
            wandb.init(
                dir=str(self.log_dir),
                name=self.run_name,
                config=dict(config) if config else None,
            )
        elif self.backend == "none":
            pass
        else:
            raise ValueError(f"unknown tracker backend {backend!r}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        elif self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def log_scalars(self, values: Mapping[str, float], step: int, prefix: str = "") -> None:
        for k, v in values.items():
            self.log_scalar(f"{prefix}{k}", float(v), step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        elif self._wandb is not None:
            self._wandb.finish()


def _as_markdown(config: Mapping[str, Any]) -> str:
    lines = ["| key | value |", "| --- | --- |"]
    for k, v in config.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
