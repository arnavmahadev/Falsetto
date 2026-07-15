"""Phase 1 data-pipeline tests (T-06 .. T-11) using synthetic audio."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from falsetto.data import (
    Augmentor,
    EmbeddingCache,
    SplitRatios,
    Stage1ClipDataset,
    assign_splits,
    build_manifest,
    class_balance,
    collate_clips,
    conform,
    load_for_extractor,
    make_dataloader,
    scan_fakemusiccaps,
    segment_fixed,
    spec_for,
    stratified_group_split,
    verify_no_leakage,
)
from falsetto.utils.audio_io import save_audio

MODELS = ["MusicGen", "MusicLDM", "AudioLDM2", "StableAudioOpen", "Mustango"]


def _tone(sr: int, seconds: float, freq: float = 220.0, channels: int = 1) -> torch.Tensor:
    t = torch.arange(int(sr * seconds)) / sr
    wave = torch.sin(2 * math.pi * freq * t).unsqueeze(0)
    return wave.repeat(channels, 1)


@pytest.fixture
def fake_dataset(tmp_path):
    """Write a tiny FakeMusicCaps-shaped tree: N tracks x (real + 5 models)."""
    root = tmp_path / "fakemusiccaps"
    sr = 32000
    n_tracks = 12
    for i in range(n_tracks):
        tid = f"ytid{i:03d}"
        save_audio(root / "real" / f"{tid}.wav", _tone(sr, 10.0, 200 + i), sr)
        for m in MODELS:
            save_audio(root / m / f"{tid}_{m.lower()}.wav", _tone(sr, 10.0, 300 + i), sr)
    return root, sr, n_tracks


# --------------------------------------------------------------------------- #
# T-07 per-extractor resampling
# --------------------------------------------------------------------------- #
def test_load_for_extractor_rates_and_channels(fake_dataset):
    root, _sr, _ = fake_dataset
    path = next((root / "real").glob("*.wav"))
    for name, exp_sr, exp_ch in [("wav2vec2", 16000, 1), ("mert", 24000, 1), ("fxencoder", 44100, 2)]:
        wav, sr = load_for_extractor(path, name)
        assert sr == exp_sr
        assert wav.shape[0] == exp_ch
        # ~10 s at the target rate
        assert abs(wav.shape[1] - exp_sr * 10) <= exp_sr * 0.02


def test_conform_mono_to_stereo():
    mono = torch.randn(1, 1000)
    out = conform(mono, 16000, spec_for("fxencoder"))
    assert out.shape[0] == 2
    # resampler length is ~ round(n * new/old), allow a 1-sample rounding slack
    assert abs(out.shape[1] - round(1000 * 44100 / 16000)) <= 1


# --------------------------------------------------------------------------- #
# T-08 fixed segmentation
# --------------------------------------------------------------------------- #
def test_segment_fixed_boundaries_and_padding():
    sr = 16000
    wave = torch.arange(sr * 25).float().unsqueeze(0)  # 25 s mono, ramp
    clips = segment_fixed(wave, sr, clip_seconds=10.0)
    # 25 s -> [0-10), [10-20), then a 5 s remainder padded to 10 s = 3 clips
    assert clips.shape == (3, 1, sr * 10)
    # Deterministic boundary: second clip starts at sample 10*sr
    assert clips[1, 0, 0].item() == float(sr * 10)
    # Padded tail is zero after the 5 s of real audio
    assert clips[2, 0, sr * 5 :].abs().sum().item() == 0.0


def test_segment_fixed_short_clip_padded():
    sr = 16000
    wave = torch.ones(1, sr * 3)  # 3 s
    clips = segment_fixed(wave, sr, clip_seconds=10.0)
    assert clips.shape == (1, 1, sr * 10)
    assert clips[0, 0, : sr * 3].sum().item() == pytest.approx(sr * 3)


# --------------------------------------------------------------------------- #
# T-09 augmentation
# --------------------------------------------------------------------------- #
def test_augmentor_preserves_length_and_is_deterministic():
    sr = 16000
    wave = _tone(sr, 2.0, 220.0)
    a1 = Augmentor(aug_prob=1.0, enabled=True, seed=0)(wave, sr)
    a2 = Augmentor(aug_prob=1.0, enabled=True, seed=0)(wave, sr)
    assert a1.shape == wave.shape  # length preserved after time-stretch
    assert torch.allclose(a1, a2)  # same seed -> same augmentation


def test_augmentor_disabled_is_identity():
    sr = 16000
    wave = _tone(sr, 1.0)
    out = Augmentor(enabled=False)(wave, sr)
    assert torch.equal(out, wave)


# --------------------------------------------------------------------------- #
# T-06 manifest + split
# --------------------------------------------------------------------------- #
def test_scan_matches_real_zenodo_layout(tmp_path):
    """The published zip's folder names and __MACOSX stubs, not the idealised ones.

    Zenodo record 15063698 ships `MusicGen_medium/` (not `MusicGen/`) and an
    __MACOSX/ tree whose AppleDouble stubs are named `._<track>.wav`, so they
    match AUDIO_EXTS and otherwise scan as real clips.
    """
    root = tmp_path / "FakeMusicCaps"
    sr = 32000
    zenodo_dirs = ["MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"]
    n_tracks = 4
    for i in range(n_tracks):
        tid = f"dHGAXJ9RPJ{i}"
        for d in zenodo_dirs:
            save_audio(root / d / f"{tid}.wav", _tone(sr, 1.0, 300 + i), sr)
            # AppleDouble stub shadowing each clip, as the zip carries it
            stub = root / "__MACOSX" / d / f"._{tid}.wav"
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_bytes(b"\x00\x05\x16\x07")

    records = scan_fakemusiccaps(root)
    assert len(records) == n_tracks * len(zenodo_dirs)
    assert not any("__MACOSX" in r["filepath"] for r in records)
    # Every generator dir is attributed, none fall back to "unknown"
    assert {r["source"] for r in records} == {d.lower() for d in zenodo_dirs}
    assert all(r["label"] == 1 for r in records)
    # One id per caption, shared across the five generators, so splits stay leak-free
    assert len({r["track_id"] for r in records}) == n_tracks


def test_scan_and_build_manifest(fake_dataset):
    root, _sr, n_tracks = fake_dataset
    records = scan_fakemusiccaps(root)
    assert len(records) == n_tracks * 6  # real + 5 models
    df = build_manifest(records, dataset="fakemusiccaps")
    assert set(df.columns) >= {"filepath", "track_id", "label", "source", "dataset", "split"}
    assert (df["label"] == 0).sum() == n_tracks  # one real per track
    assert (df["label"] == 1).sum() == n_tracks * 5
    # Each real + its 5 fakes share a track_id
    assert df["track_id"].nunique() == n_tracks


def test_split_is_leak_free_and_stratified(fake_dataset):
    root, _sr, _ = fake_dataset
    df = build_manifest(scan_fakemusiccaps(root), dataset="fakemusiccaps")
    df = assign_splits(df, SplitRatios(0.8, 0.1, 0.1), seed=1)
    verify_no_leakage(df)  # raises if a track spans splits
    bal = class_balance(df)
    assert set(bal.index) <= {"train", "val", "test"}
    # every split has both classes represented for this balanced toy set
    assert "train" in bal.index


def test_stratified_group_split_direct():
    tracks = [f"t{i}" for i in range(100) for _ in range(3)]  # 3 clips each
    labels = [i % 2 for i in range(100) for _ in range(3)]
    splits = stratified_group_split(tracks, labels, SplitRatios(0.8, 0.1, 0.1), seed=0)
    s = pd.Series(splits, index=tracks)
    # No track in two splits
    assert (s.groupby(level=0).nunique() == 1).all()
    frac = pd.Series(splits).value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.05)


# --------------------------------------------------------------------------- #
# T-10 dataset + dataloader
# --------------------------------------------------------------------------- #
def test_dataset_and_dataloader_one_epoch(fake_dataset):
    root, _sr, _ = fake_dataset
    df = assign_splits(build_manifest(scan_fakemusiccaps(root), "fakemusiccaps"), seed=2)
    ds = Stage1ClipDataset(df, extractor="mert", clip_seconds=10.0, split="train", random_crop=True)
    assert len(ds) > 0
    wav, label = ds[0]
    assert wav.shape == (1, 24000 * 10)
    assert label.dtype == torch.float32

    loader = make_dataloader(ds, batch_size=4, shuffle=True, num_workers=0)
    seen = 0
    for x, y in loader:
        assert x.shape[0] == y.shape[0]
        assert x.shape[1:] == (1, 24000 * 10)
        seen += x.shape[0]
    assert seen == len(ds)


def test_collate_pads_variable_length():
    batch = [(torch.ones(1, 100), torch.tensor(1.0)), (torch.ones(1, 250), torch.tensor(0.0))]
    x, y = collate_clips(batch)
    assert x.shape == (2, 1, 250)  # padded to longest
    assert y.tolist() == [1.0, 0.0]


# --------------------------------------------------------------------------- #
# T-11 embedding cache
# --------------------------------------------------------------------------- #
def test_embedding_cache_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    emb = torch.randn(48, 768)
    assert not cache.has("mert", "clip123")
    cache.save("mert", "clip123", emb)
    assert cache.has("mert", "clip123")
    loaded = cache.load("mert", "clip123")
    assert torch.allclose(loaded, emb)


def test_cache_get_or_compute_skips_recompute(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.zeros(4)

    a = cache.get_or_compute("mert", "x", compute)
    b = cache.get_or_compute("mert", "x", compute)  # cache hit
    assert calls["n"] == 1
    assert torch.equal(a, b)
    assert cache.clear("mert") == 1
