#!/usr/bin/env python
"""FALSETTO Studio — the interactive demo (TASKS.md T-43).

Self-contained: runs the real MERT / beat-tracking / SSM / Fusion pipeline on any
uploaded track and shows the self-similarity matrix, segments, fusion gate and a
verdict. On first run it builds the demo assets (checkpoint + example clips)
automatically; no dataset download required.

Requires the demo extra:  pip install -e ".[demo]"

    python scripts/demo.py                    # builds assets if missing, then launches
    python scripts/demo.py --assets demo_assets --share
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", default="demo_assets", help="demo assets dir (auto-built if missing)")
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = p.parse_args(argv)

    try:
        import gradio  # noqa: F401
    except ImportError:
        print('gradio not installed; run: pip install -e ".[demo]"', file=sys.stderr)
        return 1

    from falsetto.demo.studio import launch

    launch(assets_dir=args.assets, device=args.device, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
