"""Per-clip embedding cache (TASKS.md T-11).

Stage-2 re-uses Stage-1 segment embeddings many times, so we compute them once
and cache to disk as ``.pt`` tensors keyed by ``(extractor, clip_id)``. Files are
sharded by a short hash prefix to avoid huge flat directories.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch


class EmbeddingCache:
    """Filesystem cache mapping ``(extractor, clip_id) -> Tensor``."""

    def __init__(self, cache_dir: str | Path, shard_depth: int = 2) -> None:
        self.cache_dir = Path(cache_dir)
        self.shard_depth = shard_depth

    def _key(self, extractor: str, clip_id: str) -> str:
        return hashlib.sha1(f"{extractor}::{clip_id}".encode()).hexdigest()

    def path_for(self, extractor: str, clip_id: str) -> Path:
        digest = self._key(extractor, clip_id)
        shard = digest[: self.shard_depth]
        return self.cache_dir / extractor / shard / f"{digest}.pt"

    def has(self, extractor: str, clip_id: str) -> bool:
        return self.path_for(extractor, clip_id).exists()

    def save(self, extractor: str, clip_id: str, tensor: torch.Tensor) -> Path:
        """Write ``tensor`` (moved to CPU, detached) for the key. Returns the path."""
        path = self.path_for(extractor, clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename so a crash never leaves a half file.
        tmp = path.with_suffix(".pt.tmp")
        torch.save(tensor.detach().to("cpu").contiguous(), tmp)
        tmp.replace(path)
        return path

    def load(self, extractor: str, clip_id: str, map_location: str = "cpu") -> torch.Tensor:
        path = self.path_for(extractor, clip_id)
        if not path.exists():
            raise KeyError(f"cache miss for ({extractor!r}, {clip_id!r}) at {path}")
        return torch.load(path, map_location=map_location)

    def get_or_compute(
        self,
        extractor: str,
        clip_id: str,
        compute: Callable[[], torch.Tensor],
        map_location: str = "cpu",
    ) -> torch.Tensor:
        """Return the cached tensor, computing + storing it on a miss."""
        if self.has(extractor, clip_id):
            return self.load(extractor, clip_id, map_location=map_location)
        tensor = compute()
        self.save(extractor, clip_id, tensor)
        return tensor

    def clear(self, extractor: str | None = None) -> int:
        """Delete cached files (optionally for one extractor). Returns count removed."""
        base = self.cache_dir / extractor if extractor else self.cache_dir
        if not base.exists():
            return 0
        removed = 0
        for path in base.rglob("*.pt"):
            path.unlink()
            removed += 1
        return removed
