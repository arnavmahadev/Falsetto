#!/usr/bin/env python
"""Phase-0 utilities smoke test (TASKS.md T-04).

Exercises the four utility modules end to end:
  - seed everything,
  - select a device,
  - log a scalar to the experiment tracker,
  - synthesize a wav, save it, and load it back.

    python scripts/smoke_utils.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

from falsetto.utils import (
    ExperimentTracker,
    get_logger,
    load_audio,
    save_audio,
    seed_everything,
    select_device,
)


def main() -> int:
    log = get_logger("smoke")
    seed_everything(42)
    device = select_device("auto")
    log.info("device=%s", device)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Log a scalar to the tracker (TensorBoard by default).
        tracker = ExperimentTracker(backend="tensorboard", log_dir=tmp / "runs", run_name="smoke")
        for step in range(5):
            tracker.log_scalar("smoke/sine", math.sin(step), step)
        tracker.close()
        log.info("logged scalars to %s", tmp / "runs")

        # Synthesize a 1 s 440 Hz stereo tone, round-trip through disk.
        sr = 24000
        t = torch.arange(sr) / sr
        tone = torch.sin(2 * math.pi * 440 * t).unsqueeze(0).repeat(2, 1)  # [2, sr]
        wav_path = tmp / "tone.wav"
        save_audio(wav_path, tone, sr)

        waveform, loaded_sr = load_audio(wav_path)
        assert loaded_sr == sr, (loaded_sr, sr)
        assert waveform.shape[0] == 2, waveform.shape
        log.info("loaded wav: shape=%s sr=%d", tuple(waveform.shape), loaded_sr)

        mono, _ = load_audio(wav_path, sr=16000, mono=True)
        log.info("resampled mono wav: shape=%s (16 kHz)", tuple(mono.shape))

    print("smoke_utils OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
