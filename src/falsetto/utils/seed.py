"""Deterministic seeding across Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed ``random``, NumPy and torch (CPU + CUDA) for reproducibility.

    When ``deterministic`` is set, also request deterministic algorithms and
    disable cuDNN autotuning. This trades a little speed for run-to-run
    reproducibility. Returns the seed so callers can log it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this to be reproducible for some matmuls.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch without warn_only
            torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so each worker is seeded deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
