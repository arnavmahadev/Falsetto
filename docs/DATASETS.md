# Datasets — acquisition, layout, licensing (TASKS.md T-05)

Audio is **never committed** to this repo (`/data/` is git-ignored). Download each
dataset yourself and record where it lives on disk; the manifest builder
(`falsetto.data.manifests`) then indexes it into a CSV.

Recommended on-disk layout (anything works as long as the model name / `real`
appears somewhere in each file's path so the scanner can label it):

```
data/
  raw/
    fakemusiccaps/
      real/            # MusicCaps reference tracks           -> label 0
      MusicGen/        # generated                             -> label 1 (source=musicgen)
      MusicLDM/        ...
      AudioLDM2/
      StableAudioOpen/
      Mustango/
    sonics/
      real/  fake/
    aime/
      real/  ai/
  manifests/           # built CSV/Parquet (also git-ignored)
  cache/               # cached .pt embeddings
```

## FakeMusicCaps  *(primary — Stage 1)*
- ~**5,373 real** / ~**27,605 AI** 10 s clips; 5 text-to-music models
  (MusicGen, MusicLDM, AudioLDM2, Stable Audio Open, Mustango).
- Source: text-to-music deepfake detection benchmark (see README references).
- Fetch with `scripts/download_data.py` (HuggingFace snapshot) or manually, then:
  ```bash
  python scripts/download_data.py count --root data/raw/fakemusiccaps
  ```
  Expect roughly 5,373 real / 27,605 AI.

## SONICS  *(Stage 1 / Stage 2)*
- **48,090 real** / **49,074 AI**, ~176 s average; Suno + Udio. Resample to 16 kHz.
- Large; a subset is fine for this reproduction.

## AIME  *(Stage 2, harder)*
- **6,000 real** (MTG-Jamendo) / **6,000 AI**; fewer artifacts than FakeMusicCaps.
- Needs the **MTG-Jamendo** real tracks for its real class.

## Licensing
Each dataset and the MTG-Jamendo source retain **their own licenses and citation
requirements** — consult the original source before use or redistribution. This
repo bundles none of the audio. See the README "Built on the work of others"
section for citations.
