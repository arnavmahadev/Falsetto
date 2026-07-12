"""Audio I/O: load/save waveforms as torch tensors, mono/stereo aware.

Loading goes through ``soundfile`` (libsndfile) for broad format support and
falls back to ``torchaudio`` for anything it can't open (e.g. mp3 on some
builds). The canonical in-memory layout is ``[channels, samples]`` float32,
matching torchaudio. Resampling here is a general convenience; the
per-extractor rate policy lives in :mod:`falsetto.data.audio` (T-07).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def load_audio(
    path: str | Path,
    sr: Optional[int] = None,
    mono: bool = False,
) -> tuple[torch.Tensor, int]:
    """Load an audio file.

    Args:
        path: audio file path.
        sr: if given, resample to this rate; otherwise keep the file's rate.
        mono: if True, downmix to a single channel.

    Returns:
        ``(waveform, sample_rate)`` where waveform is float32 ``[channels, samples]``.
    """
    path = Path(path)
    waveform, file_sr = _read(path)

    if mono:
        waveform = to_mono(waveform)
    if sr is not None and sr != file_sr:
        waveform = resample(waveform, file_sr, sr)
        file_sr = sr
    return waveform.to(torch.float32), file_sr


def _read(path: Path) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # [samples, channels]
        waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # [channels, samples]
        return waveform, int(file_sr)
    except Exception:
        import torchaudio

        waveform, file_sr = torchaudio.load(str(path))  # already [channels, samples]
        return waveform.to(torch.float32), int(file_sr)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Downmix ``[channels, samples]`` to ``[1, samples]`` by averaging channels."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample ``[channels, samples]`` from ``orig_sr`` to ``target_sr``."""
    if orig_sr == target_sr:
        return waveform
    import torchaudio.functional as AF

    return AF.resample(waveform, orig_sr, target_sr)


def peak_normalize(waveform: torch.Tensor, peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    """Scale so the maximum absolute sample equals ``peak`` (no-op on silence)."""
    max_abs = waveform.abs().max()
    if max_abs < eps:
        return waveform
    return waveform * (peak / max_abs)


def save_audio(path: str | Path, waveform: torch.Tensor, sr: int) -> Path:
    """Write ``[channels, samples]`` (or ``[samples]``) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    import soundfile as sf

    sf.write(str(path), waveform.transpose(0, 1).cpu().numpy(), sr)
    return path
