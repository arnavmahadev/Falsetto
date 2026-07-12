#!/usr/bin/env python
"""Train a Stage-2 model on precomputed segment sequences (TASKS.md T-29 / T-35).

    python scripts/train_stage2.py --config <cfg> --seqcache data/seqcache
    # cfg.model.name selects segment_transformer (Paper 1) or fusion_segment_transformer (Paper 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seqcache", required=True, help="dir with {train,val}/ sequence .pt files")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    from falsetto.config import load_config
    from falsetto.training.train_stage2 import Stage2SequenceDataset, train_stage2_from_sequences
    from falsetto.utils.logging import get_logger

    log = get_logger("train_stage2")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.device = args.device

    seqcache = Path(args.seqcache)
    train_items = Stage2SequenceDataset.from_cache_dir(seqcache / "train").items
    val_items = Stage2SequenceDataset.from_cache_dir(seqcache / "val").items
    if not train_items:
        log.error("no training sequences in %s/train", seqcache)
        return 1
    embed_dim = train_items[0]["embeddings"].shape[-1]

    ckpt = train_stage2_from_sequences(cfg, train_items, val_items, embed_dim, cfg.data.segment_length)
    log.info("done. best checkpoint: %s", ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
