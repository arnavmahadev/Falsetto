#!/usr/bin/env python
"""Build the self-contained demo assets (checkpoint + example clips).

Generates coherent/incoherent music, extracts real MERT segment embeddings, and
trains the Fusion Segment Transformer on the structural-coherence task. Runs once;
the demo then loads the saved checkpoint.

    python scripts/build_demo.py                 # -> demo_assets/
    python scripts/build_demo.py --out demo_assets --n 24 --epochs 80
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demo_assets")
    p.add_argument("--n", type=int, default=20, help="clips per class")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from falsetto.demo.assets import build_demo_assets

    bundle = build_demo_assets(args.out, n_per_class=args.n, epochs=args.epochs,
                               seconds=args.seconds, device=args.device)
    print(f"\nDemo ready. Launch with:\n  python scripts/demo.py --assets {args.out}")
    print(f"  checkpoint: {bundle.ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
