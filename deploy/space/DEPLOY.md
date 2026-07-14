# Deploying FALSETTO Studio to Hugging Face Spaces

A one-time setup, then a single command. Result: a public URL like
`https://huggingface.co/spaces/<you>/falsetto-studio` you can put on your resume.

## Why Spaces (not GitHub Pages / Vercel)

FALSETTO Studio is a **server-side Python app** that loads PyTorch + the 95M-parameter MERT model
and runs inference. Static hosts (GitHub Pages) can't run Python at all; serverless/frontend hosts
(Vercel) cap bundle size and runtime far below what PyTorch needs. Spaces gives a long-running
container with the model warm in memory and native Gradio support.

> **Cost note (2026):** Hosting this app on a Hugging Face *personal* account now needs **PRO**
> (~$9/mo) either way. New free accounts can no longer provision **CPU Basic** for a Gradio Space
> (that's the `402` you'll hit), and the [ZeroGPU docs](https://huggingface.co/docs/hub/en/spaces-zerogpu)
> state that *hosting* your own ZeroGPU Space also requires PRO (free accounts can only *use* other
> people's ZeroGPU Spaces). No other free always-on host fits either: Render/Railway/Fly free tiers
> are ≤512 MB or need a card, and this image is ~1.5 GB (Torch + MERT). `app.py` is ready to run on
> ZeroGPU (see below) the moment an account is PRO, but see the free Colab path first.

## Free live link, no PRO: run it on Colab

The genuinely-free way to get a real, clickable demo. Colab gives a free GPU and Gradio's
`share=True` mints a public `https://…gradio.live` link that lives as long as the session
(~72 h max). Not always-on, but zero cost and it runs the real pipeline on GPU. In a Colab cell:

```python
!pip -q install "falsetto @ git+https://github.com/arnavmahadev/Falsetto.git@main" gradio
!git clone -q https://github.com/arnavmahadev/Falsetto.git   # for deploy/space/app.py + demo_assets
%cd Falsetto
import subprocess; subprocess.run(["python","scripts/build_demo.py"])  # ~2 min: MERT + example clips
import sys; sys.path.insert(0, "deploy/space")
import app                       # loads the analyzer on Colab's CUDA (spaces absent -> shim)
app.demo.launch(share=True, theme=app._theme(), css=app._CSS)
```

The same `app.py` runs unchanged: off-Spaces the `@spaces.GPU` decorator is a no-op shim and the
analyzer auto-selects Colab's real CUDA.

## One-time setup

1. Make a free account at https://huggingface.co.
2. Create a **WRITE** access token at https://huggingface.co/settings/tokens.
3. Log in from this repo's environment:

   ```bash
   .venv/bin/hf auth login      # paste the WRITE token
   ```

## Deploy

```bash
.venv/bin/python deploy/space/deploy.py <your-hf-username>/falsetto-studio
```

That creates the Space (if new), bundles `app.py`, `requirements.txt`, the Space `README.md`, and
the pre-built `demo_assets/` (model + example clips, via LFS), and uploads. First build takes
~5-10 min while it installs Torch/Transformers and caches MERT; after that, cold starts are quick.

To run on **ZeroGPU**, select it under the Space's **Settings → Hardware** (PRO), and edit
`requirements.txt` per its ZeroGPU note (drop the CPU torch index). `app.py` needs no change; it
detects ZeroGPU at runtime and moves inference onto the attached GPU automatically.

## What's in this folder

| File | Role |
|---|---|
| `app.py` | Space entrypoint: wraps inference in `@spaces.GPU` for ZeroGPU (no-op shim + auto device off-Spaces), serves the Studio UI. |
| `requirements.txt` | CPU Torch + `falsetto` installed from GitHub `main`. |
| `README.md` | Space "card" with the Gradio SDK metadata header. |
| `.gitattributes` | Tracks the model/wavs with Git LFS. |
| `deploy.py` | Creates the Space and uploads everything. |

## Updating later

Re-run the same `deploy.py` command; it pushes a new commit to the Space. Because
`requirements.txt` installs `falsetto` from GitHub `main`, pushing new code to `main` and
re-deploying picks it up. To ship only new demo assets, rebuild them (`python scripts/build_demo.py`)
and re-run deploy.
