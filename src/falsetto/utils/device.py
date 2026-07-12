"""Device selection and AMP (autocast) helpers.

Targets CUDA, Apple MPS, and CPU. ``"auto"`` prefers CUDA, then MPS, then CPU.
The autocast helper picks a dtype/enabled combination that is actually valid for
the chosen device (MPS autocast support is partial, so it is opt-in there).
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a device string to a concrete :class:`torch.device`.

    ``preference`` is one of ``auto | cpu | cuda | mps``. An explicit choice that
    is unavailable falls back to CPU with the caller free to warn.
    """
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if _mps_available() else "cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device preference {preference!r}")


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def device_report() -> dict[str, object]:
    """Human/log-friendly summary of what accelerators are available."""
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": _mps_available(),
        "selected_auto": str(select_device("auto")),
    }
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_count"] = torch.cuda.device_count()
    return report


def autocast(device: torch.device, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    """Return an autocast context manager appropriate for ``device``.

    - CUDA: bfloat16 if supported else float16.
    - CPU: bfloat16 autocast.
    - MPS: autocast is only partially supported; return a no-op unless the
      running torch exposes an MPS autocast path.
    """
    if not enabled:
        return contextlib.nullcontext()

    dtype_for = dtype
    if device.type == "cuda":
        if dtype_for is None:
            dtype_for = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype_for)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=dtype_for or torch.bfloat16)
    if device.type == "mps":
        try:
            return torch.autocast(device_type="mps", dtype=dtype_for or torch.float16)
        except (RuntimeError, ValueError):  # autocast unsupported on this build
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def amp_dtype(device: torch.device) -> torch.dtype:
    """The dtype :func:`autocast` would use for ``device`` (for GradScaler setup)."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16
