"""Hugging Face Spaces entrypoint for FALSETTO Studio.

Serves the same Gradio interface as `python scripts/demo.py`, pinned to CPU and
loading the pre-built demo model + example clips that are bundled next to this
file (so cold starts don't retrain). MERT downloads once from the Hub and is
cached by the Space.
"""

from __future__ import annotations

from pathlib import Path

from falsetto.demo.studio import _CSS, _theme, build_interface, load_analyzer


def _find_assets() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "demo_assets", here.parent.parent / "demo_assets", Path("demo_assets")):
        if (cand / "fusion_demo.pt").exists():
            return cand
    # Nothing pre-built found — load_analyzer will build assets on first run.
    return here / "demo_assets"


ASSETS = _find_assets()
analyzer, meta = load_analyzer(ASSETS, device="cpu")
demo = build_interface(analyzer, meta, ASSETS / "examples")
demo.queue(max_size=24)

if __name__ == "__main__":
    # Gradio reads GRADIO_SERVER_NAME / _PORT from the Space environment.
    demo.launch(theme=_theme(), css=_CSS)
