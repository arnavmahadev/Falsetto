#!/usr/bin/env python
"""Dataset acquisition + counting helper (TASKS.md T-05).

Audio is never committed. This script (a) points at where each dataset lives,
(b) can snapshot a HuggingFace dataset repo into ``data/raw/...``, and (c) counts
real/AI clips in a downloaded tree so you can sanity-check against the expected
totals.

    # print sources + expected counts
    python scripts/download_data.py info

    # download a HF dataset snapshot (repo id varies by mirror)
    python scripts/download_data.py download --repo <org/dataset> --dest data/raw/fakemusiccaps

    # count real vs AI in a downloaded FakeMusicCaps tree
    python scripts/download_data.py count --root data/raw/fakemusiccaps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCES = {
    "fakemusiccaps": {"real": 5373, "ai": 27605, "note": "10 s clips, 5 TTM models"},
    "sonics": {"real": 48090, "ai": 49074, "note": "~176 s avg, Suno + Udio, 16 kHz"},
    "aime": {"real": 6000, "ai": 6000, "note": "MTG-Jamendo reals; harder"},
}


def cmd_info(_: argparse.Namespace) -> int:
    print("Expected dataset composition (download separately; see docs/DATASETS.md):\n")
    for name, meta in SOURCES.items():
        print(f"  {name:16s} real={meta['real']:>6}  ai={meta['ai']:>6}   ({meta['note']})")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed; `pip install -e .`", file=sys.stderr)
        return 1
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {dest} ...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=args.allow or None,
    )
    print("Done. Now run: "
          f"python scripts/download_data.py count --root {dest}")
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    # Import here so `info` works without the package installed.
    from falsetto.data.manifests import scan_fakemusiccaps

    root = Path(args.root)
    if not root.exists():
        print(f"root not found: {root}", file=sys.stderr)
        return 1
    records = scan_fakemusiccaps(root)
    real = sum(1 for r in records if r["label"] == 0)
    ai = sum(1 for r in records if r["label"] == 1)
    print(f"{root}: {len(records)} clips  |  real={real}  ai={ai}")
    if not records:
        print("(no audio found — check the layout in docs/DATASETS.md)", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print expected dataset counts").set_defaults(func=cmd_info)

    p_dl = sub.add_parser("download", help="snapshot a HuggingFace dataset repo")
    p_dl.add_argument("--repo", required=True, help="HF dataset repo id (org/name)")
    p_dl.add_argument("--dest", required=True, help="destination directory")
    p_dl.add_argument("--allow", nargs="*", help="glob allow-patterns to limit the download")
    p_dl.set_defaults(func=cmd_download)

    p_ct = sub.add_parser("count", help="count real/AI in a FakeMusicCaps tree")
    p_ct.add_argument("--root", required=True, help="downloaded dataset root")
    p_ct.set_defaults(func=cmd_count)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
