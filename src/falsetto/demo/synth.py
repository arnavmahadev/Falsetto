"""Procedural music generator for the self-contained demo.

Produces short mono clips at 24 kHz (MERT's rate) with genuine musical structure:
chords + an arpeggio melody + a kick/hat groove, arranged into sections.

Two populations, contrasting the exact signal the Segment Transformer exploits:

- ``coherent=True``  — one key, a repeating chord progression and motif across a
  verse/chorus form, steady tempo. Strong long-range self-similarity.
- ``coherent=False`` — per-bar key/chord jumps, a reshuffled motif and tempo
  jitter. Broken structure, the way AI generations tend to drift.

No samples or external assets — everything is synthesized from oscillators.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 24000
MAJOR = [0, 2, 4, 5, 7, 9, 11]
# Common pop/classical progressions as semitone roots relative to the key.
PROGRESSIONS = [
    [0, 7, 9, 5],   # I–V–vi–IV
    [0, 9, 5, 7],   # I–vi–IV–V (50s)
    [9, 5, 0, 7],   # vi–IV–I–V
    [0, 5, 9, 7],   # I–IV–vi–V
    [2, 7, 0, 0],   # ii–V–I
]
QUALITIES = {"maj": [0, 4, 7], "min": [0, 3, 7]}
# Harmonic amplitude profiles -> different timbres (pad / organ / pluck / bell).
TIMBRES = [
    (1.0, 0.5, 0.28, 0.14),
    (1.0, 0.7, 0.5, 0.35, 0.2),
    (1.0, 0.25, 0.6, 0.1, 0.3),
    (1.0, 0.0, 0.5, 0.0, 0.33, 0.0, 0.2),
]


def _midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _adsr(n: int, sr: int, a=0.01, d=0.1, s=0.7, r=0.1) -> np.ndarray:
    env = np.ones(n)
    ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
    ai, di, ri = min(ai, n), min(di, n), min(ri, n)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:n - ri] = s
    if ri:
        env[n - ri:] = np.linspace(s, 0, ri)
    return env


def _tone(freq: float, dur: float, sr: int, harmonics=(1.0, 0.5, 0.28, 0.14), detune=0.0) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    wave = np.zeros(n)
    for i, amp in enumerate(harmonics, start=1):
        wave += amp * np.sin(2 * np.pi * freq * i * (1 + detune) * t)
    return wave * _adsr(n, sr)


def _chord_timbre(root_midi, quality, dur, sr, detune, timbre) -> np.ndarray:
    out = np.zeros(max(1, int(dur * sr)))
    for interval in QUALITIES[quality]:
        out += _tone(_midi_to_freq(root_midi + interval), dur, sr, harmonics=timbre, detune=detune) * 0.33
    return out


def _kick(dur: float, sr: int) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    f = 110 * np.exp(-t * 24) + 45  # pitch-drop
    env = np.exp(-t * 9)
    return 0.9 * np.sin(2 * np.pi * f * t) * env


def _hat(dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    return 0.25 * rng.standard_normal(n) * np.exp(-t * 60)


def generate_clip(
    seed: int,
    coherent: bool = True,
    seconds: float = 12.0,
    sr: int = SR,
) -> torch.Tensor:
    """Generate one mono clip ``[1, samples]`` as float32 in [-1, 1]."""
    rng = np.random.default_rng(seed)
    key = int(rng.integers(45, 62))  # tonic MIDI
    bpm = float(rng.uniform(84, 138))
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = max(4, int(seconds / bar))
    detune = float(rng.uniform(0.0, 0.004))
    progression = PROGRESSIONS[int(rng.integers(len(PROGRESSIONS)))]
    timbre = TIMBRES[int(rng.integers(len(TIMBRES)))]

    # A fixed melodic motif (scale-degree indices) reused when coherent.
    motif = rng.integers(0, 7, size=4)

    if coherent:
        # Render one 4-bar phrase, then tile it verbatim -> strong periodic
        # self-similarity (the hallmark of real, well-structured music).
        phrase = []
        for b in range(len(progression)):
            root = key + progression[b]
            quality = "maj" if progression[b] in (0, 5, 7) else "min"
            phrase.append(_render_bar(key, root, quality, motif, bpm, sr, detune, timbre, rng))
        bars = [phrase[b % len(phrase)] for b in range(n_bars)]
    else:
        # Every bar drifts: key jumps, new chord/motif, tempo jitter; then shuffle.
        bars = []
        for _b in range(n_bars):
            k = key + int(rng.integers(-5, 6))
            root = k + int(rng.choice([0, 2, 4, 5, 7, 9]))
            quality = str(rng.choice(["maj", "min"]))
            deg = rng.integers(0, 7, size=4)
            b_bpm = bpm * float(rng.uniform(0.85, 1.18))
            bars.append(_render_bar(k, root, quality, deg, b_bpm, sr, detune, timbre, rng))
        rng.shuffle(bars)  # disrupt long-range order

    audio = np.concatenate(bars)
    target = int(seconds * sr)
    audio = audio[:target] if len(audio) >= target else np.pad(audio, (0, target - len(audio)))
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = 0.9 * audio / peak
    return torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)


def _render_bar(key, root, quality, degrees, bpm, sr, detune, timbre, rng) -> np.ndarray:
    beat = 60.0 / bpm
    bar_len = int(beat * 4 * sr)
    out = np.zeros(bar_len)

    # Sustained chord pad under the whole bar.
    pad = _chord_timbre(root, quality, beat * 4, sr, detune, timbre)
    out[: len(pad)] += 0.5 * pad[:bar_len]

    # Arpeggio melody: one note per beat from the motif, an octave up.
    for i in range(4):
        deg = int(degrees[i % len(degrees)])
        note = key + 12 + MAJOR[deg % 7]
        start = int(i * beat * sr)
        tone = _tone(_midi_to_freq(note), beat * 0.9, sr, harmonics=timbre) * 0.4
        end = min(start + len(tone), bar_len)
        out[start:end] += tone[: end - start]

    # Groove: kick on beats, hat on offbeats.
    for i in range(4):
        s = int(i * beat * sr)
        k = _kick(beat * 0.5, sr)
        out[s: s + len(k)] += k[: bar_len - s]
        so = int((i + 0.5) * beat * sr)
        h = _hat(beat * 0.3, sr, rng)
        if so < bar_len:
            out[so: so + len(h)] += h[: bar_len - so]
    return out
