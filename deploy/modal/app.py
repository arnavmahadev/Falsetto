"""Deploy FALSETTO Studio on Modal — a free, always-reachable URL.

Modal's free **Starter** plan ($30/mo of compute credit, no credit card) serves
this as a **scale-to-zero** web endpoint: $0 while idle, a one-time cold start on
the first hit after idle, then warm for ``scaledown_window`` seconds. The MERT
weights, all Python deps, and the pre-built demo model + example clips are baked
into the image, so a cold start only *loads* the model into RAM — no downloads.

Why not always-warm (zero cold start)? ``min_containers=1`` running 24/7 is
~$90/mo of credit, over the free $30. Scale-to-zero is what keeps it truly free.

Deploy (from the repo root)::

    pip install modal
    modal setup                        # one-time browser auth — free, no card
    modal deploy deploy/modal/app.py   # prints the public https://…modal.run URL

The URL is stable across redeploys. See deploy/modal/README.md.
"""

from __future__ import annotations

from pathlib import Path

import modal

# demo_assets/ (the ~13 MB trained demo model + example clips) live at the repo
# root; bake them in at a fixed remote path.
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_LOCAL = REPO_ROOT / "demo_assets"
ASSETS_REMOTE = "/assets/demo_assets"
HF_CACHE = "/cache"  # baked HF hub cache so MERT isn't re-downloaded on cold start

app = modal.App("falsetto-studio")


def _cache_mert() -> None:
    """Runs at image-build time: downloads MERT into the baked HF cache so cold
    starts load it from disk instead of hitting the network."""
    from falsetto.config.schema import ExtractorConfig
    from falsetto.extractors import build_extractor

    build_extractor(ExtractorConfig(name="mert"))


image = (
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg + libsndfile cover audio decoding (uploads may be mp3/flac/…).
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env({"HF_HOME": HF_CACHE, "GRADIO_ANALYTICS_ENABLED": "False"})
    # CPU-only Torch keeps the image lean; Modal runs this on CPU.
    .pip_install(
        "torch>=2.2",
        "torchaudio>=2.2",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi[standard]",
        "gradio>=4",
        "falsetto @ git+https://github.com/arnavmahadev/Falsetto.git@main",
    )
    .run_function(_cache_mert)  # bake MERT into the image
    .add_local_dir(str(ASSETS_LOCAL), ASSETS_REMOTE, copy=True)
)


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,          # MERT + Torch need ~2 GB live; 4 GB is comfortable.
    scaledown_window=600,  # stay warm 10 min after the last request, then -> $0.
    # min_containers defaults to 0: scale to zero, so idle costs nothing.
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from falsetto.demo.studio import build_interface, load_analyzer

    # Loaded once per container (i.e. once per cold start), then reused.
    analyzer, meta = load_analyzer(ASSETS_REMOTE, device="cpu")
    demo = build_interface(analyzer, meta, Path(ASSETS_REMOTE) / "examples")
    demo.queue(max_size=20)
    # Theme/CSS ride along on the Blocks (set in build_interface), so mounting
    # preserves the styling without calling demo.launch().
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
