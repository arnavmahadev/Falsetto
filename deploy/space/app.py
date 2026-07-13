"""Hugging Face Spaces entrypoint for FALSETTO Studio.

Serves the same Gradio interface as ``python scripts/demo.py``. On **ZeroGPU**
(Hugging Face's dynamically-allocated GPU runtime) the analysis step is wrapped
in ``@spaces.GPU``, which attaches an NVIDIA GPU only while inference runs and
releases it afterwards. Per the ZeroGPU guide the model is placed on ``cuda`` at
module level — a CUDA emulation is active outside ``@spaces.GPU`` functions, so
this is safe and is the most efficient placement.

Off-Spaces (local run, a plain CPU Space, or Colab) the ``spaces`` package is
absent, so a no-op shim keeps the *exact same code* running on the best local
device — CPU on a CPU Space, MPS on a Mac, real CUDA on Colab. That means this
file also drives a free Colab GPU ``demo.launch(share=True)`` link unchanged.

The demo model + example clips are bundled next to this file, so cold starts
don't retrain; MERT downloads once from the Hub and is cached by the Space.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

from falsetto.demo.studio import _CSS, _theme, build_interface, load_analyzer

# `spaces` is preinstalled on ZeroGPU Spaces and is documented as effect-free
# elsewhere. When it isn't installed at all (local dev / Colab), fall back to an
# identity decorator that supports both `@spaces.GPU` and `@spaces.GPU(...)`.
try:  # pragma: no cover - import path depends on the runtime
    import spaces  # type: ignore

    _HAVE_SPACES = True
except ImportError:
    def _identity_gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]  # bare @spaces.GPU
        return lambda fn: fn  # @spaces.GPU(duration=...)

    spaces = types.SimpleNamespace(GPU=_identity_gpu)
    _HAVE_SPACES = False


def _find_assets() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "demo_assets", here.parent.parent / "demo_assets", Path("demo_assets")):
        if (cand / "fusion_demo.pt").exists():
            return cand
    # Nothing pre-built found — load_analyzer will build assets on first run.
    return here / "demo_assets"


ASSETS = _find_assets()

# On ZeroGPU, torch.cuda.is_available() is True (CUDA emulation) and models
# should sit on cuda from the start. Everywhere else, auto-pick: cpu on a plain
# CPU Space, mps on a Mac, real cuda on Colab.
_ON_ZEROGPU = _HAVE_SPACES and torch.cuda.is_available()
analyzer, meta = load_analyzer(ASSETS, device="cuda" if _ON_ZEROGPU else "auto")


@spaces.GPU(duration=120)
def _analyze(waveform, sr):
    return analyzer.analyze(waveform, sr)


demo = build_interface(analyzer, meta, ASSETS / "examples", analyze_fn=_analyze)
demo.queue(max_size=24)

if __name__ == "__main__":
    # Gradio reads GRADIO_SERVER_NAME / _PORT from the Space environment. On
    # Colab, pass share=True to get a free public gradio.live link.
    demo.launch(theme=_theme(), css=_CSS)
