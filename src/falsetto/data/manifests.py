"""Manifest builder + leak-free stratified splitter (TASKS.md T-06).

A manifest is one row per audio clip with the columns:

    filepath, track_id, label (0=real, 1=ai), source (model name),
    dataset, duration_sec, split

The split is **stratified 8:1:1** by label while keeping every clip that shares a
``track_id`` in the *same* split — so a track never appears in two splits (no
clip-level leakage). FakeMusicCaps shares a base id across a real track and its
per-model generations, so grouping by ``track_id`` also stops the real/fake pair
of one track from straddling the train/test boundary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = [
    "filepath",
    "track_id",
    "label",
    "source",
    "dataset",
    "duration_sec",
    "split",
]

# Real vs. the five FakeMusicCaps text-to-music generators.
FAKEMUSICCAPS_MODELS = {
    "musicgen",
    "musicldm",
    "audioldm2",
    "stableaudioopen",
    "stable_audio_open",
    "mustango",
}
_REAL_ALIASES = {"real", "musiccaps", "reference", "ground_truth", "gt"}


@dataclass
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def normalized(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("split ratios must sum to > 0")
        return SplitRatios(self.train / total, self.val / total, self.test / total)


def stratified_group_split(
    track_ids: "pd.Series | np.ndarray | list",
    labels: "pd.Series | np.ndarray | list",
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> np.ndarray:
    """Assign each row to ``train|val|test``, grouped by ``track_id`` and stratified by label.

    All rows sharing a ``track_id`` land in the same split. Within each label
    class the *groups* are shuffled (seeded) and partitioned by the ratios, so
    class balance is preserved across splits without any track leakage.

    Returns an array of split strings aligned to the input rows.
    """
    track_ids = np.asarray(track_ids)
    labels = np.asarray(labels)
    ratios = ratios.normalized()
    rng = np.random.default_rng(seed)

    # One representative label per group (majority vote; ties -> smallest label).
    group_rows: dict = defaultdict(list)
    for idx, tid in enumerate(track_ids):
        group_rows[tid].append(idx)
    group_label: dict = {}
    for tid, rows in group_rows.items():
        vals, counts = np.unique(labels[rows], return_counts=True)
        group_label[tid] = int(vals[np.argmax(counts)])

    # Bucket groups by their label, shuffle, and slice by cumulative ratio.
    assignment: dict = {}
    by_label: dict = defaultdict(list)
    for tid, lab in group_label.items():
        by_label[lab].append(tid)

    for lab, groups in by_label.items():
        groups = np.array(sorted(groups, key=str))  # deterministic pre-shuffle order
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * ratios.train))
        n_val = int(round(n * ratios.val))
        # Guarantee val/test get at least one group when the class is large enough.
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, n_val)
        for i, tid in enumerate(groups):
            if i < n_train:
                assignment[tid] = "train"
            elif i < n_train + n_val:
                assignment[tid] = "val"
            else:
                assignment[tid] = "test"

    return np.array([assignment[tid] for tid in track_ids])


def build_manifest(records: list[dict], dataset: str) -> pd.DataFrame:
    """Assemble a manifest DataFrame from raw records (no split assigned yet)."""
    df = pd.DataFrame.from_records(records)
    if "dataset" not in df:
        df["dataset"] = dataset
    for col in ("source", "duration_sec"):
        if col not in df:
            df[col] = None
    if "split" not in df:
        df["split"] = "unassigned"
    missing = {"filepath", "track_id", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"records missing required fields: {sorted(missing)}")
    return df[MANIFEST_COLUMNS]


def assign_splits(
    df: pd.DataFrame,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``split`` column filled in."""
    out = df.copy()
    out["split"] = stratified_group_split(out["track_id"], out["label"], ratios, seed)
    return out


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any ``track_id`` appears in more than one split."""
    per_track = df.groupby("track_id")["split"].nunique()
    leaked = per_track[per_track > 1]
    if len(leaked):
        raise AssertionError(f"track(s) leaked across splits: {list(leaked.index[:10])}")


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of real/ai per split (for printing after a build)."""
    tab = (
        df.assign(cls=df["label"].map({0: "real", 1: "ai"}))
        .groupby(["split", "cls"])
        .size()
        .unstack(fill_value=0)
    )
    tab["total"] = tab.sum(axis=1)
    return tab


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a manifest to CSV or Parquet (by file extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest from CSV or Parquet."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Dataset scanners
# --------------------------------------------------------------------------- #

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _classify_fakemusiccaps(parts: tuple[str, ...]) -> tuple[int, str]:
    """Infer (label, source) from the path components of a FakeMusicCaps file."""
    lowered = [p.lower().replace("-", "_") for p in parts]
    for p in lowered:
        if p in _REAL_ALIASES:
            return 0, "real"
    for p in lowered:
        if p in FAKEMUSICCAPS_MODELS:
            return 1, p
    # Fallback: unknown provenance is treated as AI-generated with source=unknown.
    return 1, "unknown"


def scan_fakemusiccaps(root: str | Path) -> list[dict]:
    """Walk a FakeMusicCaps download and produce manifest records.

    Assumes a layout where the model name (or ``real``/``MusicCaps``) appears as a
    path component. ``track_id`` is the file stem stripped of a model suffix, so
    the same underlying caption/track groups across models.
    """
    root = Path(root)
    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(root)
        label, source = _classify_fakemusiccaps(rel.parts)
        track_id = _base_track_id(path.stem)
        records.append(
            {
                "filepath": str(path),
                "track_id": track_id,
                "label": label,
                "source": source,
                "dataset": "fakemusiccaps",
                "duration_sec": None,
            }
        )
    return records


def _base_track_id(stem: str) -> str:
    """Strip a trailing ``_<model>`` suffix so real/fake variants share a track id."""
    lowered = stem.lower()
    for model in sorted(FAKEMUSICCAPS_MODELS, key=len, reverse=True):
        for sep in ("_", "-", "."):
            suffix = f"{sep}{model}"
            if lowered.endswith(suffix):
                return stem[: -len(suffix)]
    return stem
