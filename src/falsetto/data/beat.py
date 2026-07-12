"""Downbeat tracking (TASKS.md T-24).

Prefers **Beat This!** (`beat_this`, Foscarin et al. 2024) for downbeats; if that
optional package isn't installed, falls back to a librosa beat tracker with a
downbeat heuristic (assume 4/4, every 4th beat is a downbeat). Both return a list
of downbeat timestamps in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils.audio_io import to_mono
from ..utils.logging import get_logger

_log = get_logger("data.beat")


@dataclass
class BeatResult:
    beats: list[float]  # all beat times (seconds)
    downbeats: list[float]  # downbeat times (seconds)
    backend: str


class BeatTracker:
    """Downbeat tracker with a Beat This! -> librosa fallback."""

    def __init__(self, prefer: str = "beat_this", device: str = "cpu") -> None:
        self.prefer = prefer
        self.device = device
        self._beat_this = None
        self.backend = self._init_backend()

    def _init_backend(self) -> str:
        if self.prefer == "beat_this":
            try:
                from beat_this.inference import File2Beats

                self._beat_this = File2Beats(device=self.device)
                return "beat_this"
            except Exception as exc:  # ImportError or model download failure
                _log.warning("beat_this unavailable (%s); falling back to librosa", type(exc).__name__)
        return "librosa"

    def track(self, waveform: torch.Tensor, sample_rate: int) -> BeatResult:
        """Return beats + downbeats for a waveform ``[channels, samples]`` / ``[samples]``."""
        mono = to_mono(waveform).squeeze(0).cpu().numpy().astype(np.float32)
        if self.backend == "beat_this" and self._beat_this is not None:
            return self._track_beat_this(mono, sample_rate)
        return self._track_librosa(mono, sample_rate)

    def _track_beat_this(self, mono: np.ndarray, sr: int) -> BeatResult:
        # File2Beats also exposes an audio interface via the underlying Audio2Beats.
        beats, downbeats = self._beat_this(mono, sr)  # type: ignore[misc]
        return BeatResult(list(map(float, beats)), list(map(float, downbeats)), "beat_this")

    def _track_librosa(self, mono: np.ndarray, sr: int) -> BeatResult:
        import librosa

        _tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        # 4/4 heuristic: every 4th beat is a downbeat.
        downbeats = beat_times[::4]
        return BeatResult([float(b) for b in beat_times], [float(d) for d in downbeats], "librosa")


def get_downbeats(waveform: torch.Tensor, sample_rate: int, tracker: BeatTracker | None = None) -> list[float]:
    """Convenience: downbeat timestamps (seconds) for a track."""
    tracker = tracker or BeatTracker()
    return tracker.track(waveform, sample_rate).downbeats
