#!/usr/bin/env python
"""Train a Stage-1 detector from a config (TASKS.md T-22 entry point).

    python scripts/train_stage1.py --config configs/stage1_mert_fakemusiccaps.yaml
    python scripts/train_stage1.py --config <cfg> --epochs 5 --device cpu
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment YAML config")
    parser.add_argument("--epochs", type=int, help="override number of epochs")
    parser.add_argument("--device", help="override device (cpu/cuda/mps)")
    parser.add_argument("--manifest", help="override manifest path")
    args = parser.parse_args(argv)

    from falsetto.config import load_config
    from falsetto.training.train_stage1 import train_stage1_from_config
    from falsetto.utils.logging import get_logger

    log = get_logger("train")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.device = args.device
    if args.manifest:
        cfg.data.manifest = args.manifest

    ckpt = train_stage1_from_config(cfg)
    log.info("done. best checkpoint: %s", ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
