"""Stage-1 dataset + dataloader (TASKS.md T-10).

:class:`Stage1ClipDataset` yields ``(waveform, label)`` for fixed-length clips,
loading each file conformed to the target extractor's rate/channels. Training
uses a random crop (and optional SSL augmentation); eval uses a deterministic
leading crop. :func:`collate_clips` crops/pads a batch to a common length so
variable-length inputs stack cleanly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import seed_worker
from .audio import load_for_extractor, spec_for
from .augment import Augmentor
from .segment_fixed import _pad_to


class Stage1ClipDataset(Dataset):
    """Fixed-length clips for Stage-1 training/eval."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        extractor: str,
        clip_seconds: float = 10.0,
        split: Optional[str] = None,
        augmentor: Optional[Augmentor] = None,
        random_crop: bool = False,
        normalize: bool = False,
        seed: int = 42,
    ) -> None:
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.df = manifest.reset_index(drop=True)
        self.extractor = extractor
        self.spec = spec_for(extractor)
        self.clip_seconds = clip_seconds
        self.clip_samples = int(round(clip_seconds * self.spec.sample_rate))
        self.augmentor = augmentor
        self.random_crop = random_crop
        self.normalize = normalize
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        waveform, _sr = load_for_extractor(row["filepath"], self.extractor, normalize=self.normalize)
        waveform = self._crop_or_pad(waveform)
        if self.augmentor is not None:
            waveform = self.augmentor(waveform, self.spec.sample_rate)
            waveform = self._crop_or_pad(waveform)  # augmentation preserves length, but be safe
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return waveform, label

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        total = waveform.size(-1)
        if total < self.clip_samples:
            return _pad_to(waveform, self.clip_samples)
        if total == self.clip_samples:
            return waveform
        if self.random_crop:
            start = int(self._rng.integers(0, total - self.clip_samples + 1))
        else:
            start = 0
        return waveform[:, start : start + self.clip_samples].contiguous()


def collate_clips(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    target_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate ``(waveform, label)`` items into ``([B, C, T], [B])``.

    All waveforms are cropped/padded to ``target_samples`` (default: the longest
    in the batch), so variable-length inputs stack without error.
    """
    waveforms, labels = zip(*batch, strict=True)
    if target_samples is None:
        target_samples = max(w.size(-1) for w in waveforms)
    fitted = [_pad_to(w if w.dim() == 2 else w.unsqueeze(0), target_samples) for w in waveforms]
    return torch.stack(fitted, dim=0), torch.stack(labels, dim=0)


def make_dataloader(
    dataset: Stage1ClipDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_samples: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader wired to :func:`collate_clips` and deterministic workers."""
    if target_samples is None:
        target_samples = dataset.clip_samples
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=lambda b: collate_clips(b, target_samples=target_samples),
        drop_last=False,
    )
