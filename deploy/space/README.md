---
title: FALSETTO Studio
emoji: 🎼
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: See what an AI-music detector sees — live self-similarity structure via MERT.
---

# FALSETTO Studio

An interactive **structure explorer** for AI-generated-music detection — a reproduction of the
two-stage *Segment Transformer* (Kim & Go, [arXiv:2509.08283](https://arxiv.org/abs/2509.08283) /
[arXiv:2601.13647](https://arxiv.org/abs/2601.13647)).

Upload a track (or pick a bundled example) and the app runs the **real** pipeline on CPU:
MERT embeddings of beat-tracked segments, a live **self-similarity matrix**, the segmentation,
and the Fusion model's **per-segment gate**. Structured music shows repeated blocks/diagonals in
the SSM; drifting or unstructured audio doesn't.

> The structure estimate comes from a small **proxy-trained** demo model (structured vs. drifting
> audio) and is illustrative — not a validated AI-vs-real verdict. The visualizations, though, are
> computed for real on your audio.

Code: **https://github.com/arnavmahadev/Falsetto**
