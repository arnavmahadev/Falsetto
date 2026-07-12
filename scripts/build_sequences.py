#!/usr/bin/env python
"""Precompute Stage-2 segment-embedding sequences (TASKS.md T-26).

For each track in a manifest: beat-track -> 4-bar segments -> Stage-1 embed each
-> pad/crop to N=48 -> save ``{embeddings, mask, label}`` to a per-split cache dir
that ``scripts/train_stage2.py`` then trains on.

    python scripts/build_sequences.py --config <cfg> --stage1-ckpt <ckpt> --out data/seqcache
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:120]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1-ckpt", required=True, help="trained Stage-1 checkpoint")
    parser.add_argument("--out", required=True, help="output sequence cache dir")
    parser.add_argument("--manifest", help="override manifest path")
    parser.add_argument("--split", help="only build this split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    from falsetto.config import load_config
    from falsetto.data.audio import load_for_extractor, spec_for
    from falsetto.data.beat import BeatTracker
    from falsetto.data.manifests import load_manifest
    from falsetto.data.segment_sequence import SegmentSequenceBuilder, stage1_embedding_fn
    from falsetto.models.stage1 import build_stage1_model
    from falsetto.utils.device import select_device
    from falsetto.utils.logging import get_logger

    log = get_logger("build_seq")
    cfg = load_config(args.config)
    device = select_device(args.device or cfg.device)

    model = build_stage1_model(cfg).to(device)
    state = torch.load(args.stage1_ckpt, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.eval()

    sr = spec_for(cfg.extractor.name).sample_rate
    builder = SegmentSequenceBuilder(
        embed_fn=stage1_embedding_fn(model, device),
        sample_rate=sr,
        beat_tracker=BeatTracker(device=str(device)),
        max_segments=cfg.data.segment_length,
    )

    df = load_manifest(args.manifest or cfg.data.manifest)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)

    out_root = Path(args.out)
    built = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="sequences"):
        waveform, _ = load_for_extractor(row["filepath"], cfg.extractor.name)
        E, mask = builder.build(waveform, sr)
        split_dir = out_root / str(row.get("split", "all"))
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": E.cpu(), "mask": mask.cpu(), "label": float(row["label"]),
             "track_id": row["track_id"]},
            split_dir / f"{_safe(str(row['track_id']))}.pt",
        )
        built += 1
    log.info("built %d sequences -> %s", built, out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
