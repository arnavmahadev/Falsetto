#!/usr/bin/env python
"""Build a training manifest from a downloaded dataset tree (TASKS.md T-06 helper).

Scans a FakeMusicCaps-style folder (a ``real/`` dir plus per-model dirs, or any
tree where ``real`` / a generator name appears in each file's path), optionally
subsamples a balanced set, assigns a leak-free 8:1:1 split, and writes a CSV.

    python scripts/build_manifest.py --root data/raw/fakemusiccaps \
        --out data/manifests/fakemusiccaps.csv

    # balanced small subset for a quick GPU run
    python scripts/build_manifest.py --root data/raw/fakemusiccaps \
        --out data/manifests/fakemusiccaps_small.csv --max-per-class 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="downloaded dataset root")
    p.add_argument("--out", required=True, help="manifest CSV/Parquet path to write")
    p.add_argument("--dataset", default="fakemusiccaps", help="dataset name tag")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="cap clips per class for a balanced subset (0 = keep all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from falsetto.data.manifests import (
        assign_splits,
        build_manifest,
        class_balance,
        save_manifest,
        scan_fakemusiccaps,
        verify_no_leakage,
    )

    root = Path(args.root)
    if not root.exists():
        print(f"root not found: {root}", file=sys.stderr)
        return 1

    records = scan_fakemusiccaps(root)
    if not records:
        print("no audio found — check the layout in docs/DATASETS.md", file=sys.stderr)
        return 1
    df = build_manifest(records, dataset=args.dataset)

    if args.max_per_class > 0:
        import pandas as pd

        parts = [
            g.sample(min(len(g), args.max_per_class), random_state=args.seed)
            for _, g in df.groupby("label")
        ]
        df = pd.concat(parts).reset_index(drop=True)

    df = assign_splits(df, seed=args.seed)
    verify_no_leakage(df)
    out = save_manifest(df, args.out)

    print(f"wrote {len(df)} clips -> {out}")
    print(class_balance(df).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
