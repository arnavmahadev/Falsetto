# Deploying FALSETTO Studio to Hugging Face Spaces

A one-time setup, then a single command. Result: a public URL like
`https://huggingface.co/spaces/<you>/falsetto-studio` you can put on your resume.

## Why Spaces (not GitHub Pages / Vercel)

FALSETTO Studio is a **server-side Python app** — it loads PyTorch + the 95M-parameter MERT model
and runs inference. Static hosts (GitHub Pages) can't run Python at all; serverless/frontend hosts
(Vercel) cap bundle size and runtime far below what PyTorch needs. Spaces gives a long-running
container with the model warm in memory and native Gradio support.

> **Cost note:** Hugging Face now requires a **PRO** account (~$9/mo) to host Gradio (non-static)
> Spaces; only static Spaces are free. Everything here is ready to push the moment your account is
> PRO — or point the same `app.py` at another container host (Railway/Fly) if you'd rather not.

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
~5–10 min while it installs Torch/Transformers and caches MERT; after that, cold starts are quick.

## What's in this folder

| File | Role |
|---|---|
| `app.py` | Space entrypoint — loads bundled assets on CPU, serves the Studio UI. |
| `requirements.txt` | CPU Torch + `falsetto` installed from GitHub `main`. |
| `README.md` | Space "card" with the Gradio SDK metadata header. |
| `.gitattributes` | Tracks the model/wavs with Git LFS. |
| `deploy.py` | Creates the Space and uploads everything. |

## Updating later

Re-run the same `deploy.py` command — it pushes a new commit to the Space. Because
`requirements.txt` installs `falsetto` from GitHub `main`, pushing new code to `main` and
re-deploying picks it up. To ship only new demo assets, rebuild them (`python scripts/build_demo.py`)
and re-run deploy.
