#!/usr/bin/env python
"""Batch embedding extraction (TASKS.md T-17).

Run a feature extractor over a manifest and write one embedding per clip to the
disk cache (keyed by ``(extractor, clip_id)``). Stage-2 later re-uses these.

    python scripts/extract_embeddings.py --config configs/stage1_mert_fakemusiccaps.yaml
    python scripts/extract_embeddings.py --config <cfg> --split train --limit 100

``clip_id`` defaults to each row's ``filepath`` (stable + unique). Re-running skips
clips already present in the cache unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment YAML config")
    parser.add_argument("--manifest", help="override manifest path from config")
    parser.add_argument("--split", help="only extract this split (train/val/test)")
    parser.add_argument("--limit", type=int, help="cap number of clips (debug)")
    parser.add_argument("--device", help="override device (cpu/cuda/mps)")
    parser.add_argument("--overwrite", action="store_true", help="recompute cached clips")
    args = parser.parse_args(argv)

    from falsetto.config import load_config
    from falsetto.data.cache import EmbeddingCache
    from falsetto.data.manifests import load_manifest
    from falsetto.extractors import build_extractor
    from falsetto.utils.device import select_device
    from falsetto.utils.logging import get_logger

    log = get_logger("extract")
    cfg = load_config(args.config)
    device = select_device(args.device or cfg.device)
    manifest_path = args.manifest or cfg.data.manifest

    df = load_manifest(manifest_path)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        log.error("no rows to extract from %s (split=%s)", manifest_path, args.split)
        return 1

    log.info("building extractor %r on %s", cfg.extractor.name, device)
    extractor = build_extractor(cfg.extractor).to(device)
    extractor.freeze()
    cache = EmbeddingCache(cfg.extractor.cache_dir)

    from falsetto.data.audio import load_for_extractor

    done = skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        clip_id = str(row["filepath"])
        if not args.overwrite and cache.has(cfg.extractor.name, clip_id):
            skipped += 1
            continue
        waveform, _sr = load_for_extractor(row["filepath"], cfg.extractor.name)
        emb = extractor.extract(waveform.to(device))
        cache.save(cfg.extractor.name, clip_id, emb)
        done += 1

    log.info("extracted=%d skipped=%d cache_dir=%s", done, skipped, cfg.extractor.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
