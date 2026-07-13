# Deploying FALSETTO Studio on Modal (free, no credit card)

A genuinely free way to get a **permanent, always-reachable URL** for the Studio
demo — no PRO subscription, no credit card, no server to babysit.

## Why Modal

[Modal](https://modal.com)'s free **Starter** plan includes **$30/month of compute
credit and needs no credit card**. If you ever exceed it without adding billing,
workloads simply stop instead of charging you. This app is deployed as a
**scale-to-zero** web endpoint, so:

- **$0 while idle** — you only spend credit when someone is actually using it.
- A low-traffic portfolio demo uses a few container-hours a month at most, well
  inside the $30 — effectively **free indefinitely**.
- The tradeoff is a **one-time cold start** (~15–30 s) for the first visitor
  after the app has been idle; it then stays warm for 10 minutes so subsequent
  clicks are instant. Always-warm (zero cold start) would cost ~$90/mo, over the
  free credit — so scale-to-zero is what keeps it free.

The image bakes in the MERT weights, all deps, and the pre-built demo model +
clips, so a cold start only *loads* the model into RAM — it never re-downloads.

## Deploy

From the repo root:

```bash
pip install modal
modal setup                        # one-time browser login — free, no card
modal deploy deploy/modal/app.py   # builds the image, prints the public URL
```

The first deploy takes a few minutes (it installs Torch/Transformers and caches
MERT into the image). It prints a stable `https://<you>--falsetto-studio-ui.modal.run`
URL that survives redeploys. Push new code to GitHub `main` and re-run
`modal deploy` to update (the image installs `falsetto` from `main`).

## Tuning

Edit the decorators in [`app.py`](app.py):

| Knob | Default | Effect |
|---|---|---|
| `scaledown_window` | `600` | Seconds to stay warm after the last request before scaling to $0. Longer = fewer cold starts, slightly more credit spent. |
| `min_containers` | `0` (implicit) | Set to `1` to eliminate cold starts entirely — but this runs 24/7 (~$90/mo, beyond the free credit). |
| `cpu` / `memory` | `2.0` / `4096` | Container size. MERT + Torch need ~2 GB live; 4 GB is comfortable headroom. |
| `max_inputs` | `20` | Concurrent requests one container handles. |

## Local check

`app.py` is a standard Gradio Studio build on CPU — the same interface as
`python scripts/demo.py`. To sanity-check the interface locally without Modal,
run the repo's demo (`python scripts/demo.py`); the Modal wrapper only changes
*where* it runs, not the app itself.
```
