"""Real royalty-free music for the demo (via librosa's example corpus).

These are genuine recordings (Brahms, Tchaikovsky, and CC-licensed tracks by
Kevin MacLeod et al.) bundled/downloaded by ``librosa.example`` — used both as
demo examples and, crucially, as *training* material so the coherence detector
generalizes to real audio instead of only the synthetic clips.

For the AI-like / incoherent class we take a real clip and **shuffle its
structure**: chop it into short chunks and reorder them, destroying long-range
repetition while preserving local timbre — the structural failure mode the
Segment Transformer keys on.
"""

from __future__ import annotations

import numpy as np
import torch

from .synth import SR

# librosa example ids that are actual music (excludes speech/monophonic samples).
REAL_TRACKS = ["brahms", "nutcracker", "vibeace", "sweetwaltz", "pistachio", "choice"]

# Friendly names for the examples gallery.
TRACK_TITLES = {
    "brahms": "Brahms — Hungarian Dance No. 5",
    "nutcracker": "Tchaikovsky — Sugar Plum Fairy",
    "vibeace": "Kevin MacLeod — Vibe Ace",
    "sweetwaltz": "Sweet Waltz",
    "pistachio": "Pistachio Ragtime",
    "choice": "Choice (drum & bass)",
}


def load_real_track(name: str, sr: int = SR) -> torch.Tensor:
    """Load a librosa example track, mono, resampled to ``sr`` -> ``[1, samples]``."""
    import librosa

    y, _ = librosa.load(librosa.example(name), sr=sr, mono=True)
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


def slice_clips(
    waveform: torch.Tensor,
    sr: int,
    seconds: float,
    max_clips: int,
    skip_start: float = 1.0,
    stride_frac: float = 1.0,
) -> list[torch.Tensor]:
    """Cut ``seconds``-long clips (``stride_frac`` < 1 gives overlap -> more clips)."""
    n = int(seconds * sr)
    hop = max(1, int(seconds * stride_frac * sr))
    start = int(skip_start * sr)
    clips = []
    total = waveform.size(1)
    while start + n <= total and len(clips) < max_clips:
        clips.append(waveform[:, start:start + n].contiguous())
        start += hop
    return clips


def real_music_clips(
    seconds: float,
    per_track: int = 4,
    stride_frac: float = 0.6,
    tracks: list[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Intact real-music clips (overlapping) as ``(title, waveform)`` — the structured class."""
    tracks = tracks or REAL_TRACKS
    out: list[tuple[str, torch.Tensor]] = []
    for name in tracks:
        wav = load_real_track(name)
        for j, clip in enumerate(slice_clips(wav, SR, seconds, per_track, stride_frac=stride_frac)):
            out.append((f"{TRACK_TITLES.get(name, name)} · pt{j + 1}", clip))
    return out


def shuffle_structure(
    clip: torch.Tensor,
    sr: int,
    chunk_seconds: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Chop into ``chunk_seconds`` chunks and reorder them (structure destroyed)."""
    rng = np.random.default_rng(seed)
    n = int(chunk_seconds * sr)
    total = clip.size(1)
    chunks = [clip[:, i:i + n] for i in range(0, total, n)]
    chunks = [c for c in chunks if c.size(1) > 0]
    order = rng.permutation(len(chunks))
    return torch.cat([chunks[i] for i in order], dim=1)[:, :total].contiguous()


def build_real_population(
    seconds: float,
    max_clips_per_track: int = 3,
    tracks: list[str] | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Return (human_clips, ai_like_clips) as ``(title, waveform)`` lists.

    ``human_clips`` are intact real music; ``ai_like_clips`` are the same clips
    with their structure shuffled.
    """
    tracks = tracks or REAL_TRACKS
    human: list[tuple[str, torch.Tensor]] = []
    ai_like: list[tuple[str, torch.Tensor]] = []
    for name in tracks:
        wav = load_real_track(name)
        for j, clip in enumerate(slice_clips(wav, SR, seconds, max_clips_per_track)):
            title = TRACK_TITLES.get(name, name)
            human.append((f"{title} · pt{j + 1}", clip))
            ai_like.append((f"{title} (shuffled) · pt{j + 1}",
                            shuffle_structure(clip, SR, seed=hash((name, j)) % 10_000)))
    return human, ai_like
