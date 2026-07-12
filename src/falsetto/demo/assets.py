"""Build self-contained demo assets (checkpoint + example clips).

Trains the Fusion Segment Transformer on a **structural-coherence** task over both
synthetic and *real* music, so the demo generalizes to real uploads:

  human (0)  = intact real music        + synthetic verbatim-loop clips
  AI-like (1) = structure-shuffled real music + synthetic bar-shuffled clips

MERT segment embeddings + adaptive SSMs are extracted once per clip; the Fusion
head then trains on the cached sequences. Everything is bundled — no dataset
download required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config.schema import ExtractorConfig
from ..extractors import build_extractor
from ..models.fusion import FusionSegmentTransformer
from ..training.losses import BCEWithLogits
from ..training.metrics import compute_metrics
from ..utils.audio_io import save_audio
from ..utils.device import select_device
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .pipeline import SEQ_LEN, DemoPipeline
from .realmusic import real_music_clips
from .synth import SR, generate_clip


def _noise_clip(seconds: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    # pink-ish noise: unstructured audio anchoring the low-structure class
    y = rng.standard_normal(n).astype(np.float32)
    y = np.cumsum(y); y = y / (np.abs(y).max() + 1e-6) * 0.6
    return torch.from_numpy(y).unsqueeze(0)

_log = get_logger("demo.assets")

CKPT_NAME = "fusion_demo.pt"
META_NAME = "demo_meta.json"
EMBED_DIM = 768
D_MODEL = 128


@dataclass
class DemoBundle:
    ckpt_path: Path
    meta_path: Path
    examples_dir: Path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60]


def _labeled_clips(n_synth_per_class: int, seconds: float, real_per_track: int, seed: int):
    """Assemble ``(label, source, title, waveform)`` records.

    Structured class (0): real recordings + synthetic loops.
    Unstructured class (1): synthetic drift (key-jumping) + noise — audio that
    lacks the repeated structure produced music has.
    """
    records = []
    # Structured class (0): real recordings + synthetic loops.
    for title, wav in real_music_clips(seconds, per_track=real_per_track, stride_frac=0.5):
        records.append((0, "real", title, wav))
    for i in range(n_synth_per_class):
        records.append((0, "synth", f"Synthetic loop {i + 1}",
                        generate_clip(seed=seed + i, coherent=True, seconds=seconds)))
    # Unstructured class (1): synthetic drift (same timbre, no repetition) + noise.
    for i in range(n_synth_per_class):
        records.append((1, "synth", f"Synthetic drift {i + 1}",
                        generate_clip(seed=seed + 5000 + i, coherent=False, seconds=seconds)))
    for i in range(max(3, n_synth_per_class // 4)):
        records.append((1, "noise", f"Unstructured noise {i + 1}",
                        _noise_clip(seconds, seed=seed + 9000 + i)))
    return records


def build_demo_assets(
    out_dir: str | Path,
    n_per_class: int = 16,
    epochs: int = 40,
    seconds: float = 16.0,
    real_per_track: int = 3,
    device: str = "auto",
    seed: int = 0,
) -> DemoBundle:
    out_dir = Path(out_dir)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    dev = select_device(device)
    _log.info("building demo assets on %s (runs MERT once per clip)...", dev)

    mert = build_extractor(ExtractorConfig(name="mert")).to(dev)
    mert.freeze()
    pipeline = DemoPipeline(mert, dev, seq_len=SEQ_LEN)

    records = _labeled_clips(n_per_class, seconds, real_per_track, seed)
    _log.info("embedding %d clips (%d synth + real music, both classes)...",
              len(records), 2 * n_per_class)

    feats = []
    for label, source, title, wav in records:
        f = pipeline.features(wav, SR)
        feats.append({"embeddings": f.embeddings, "ssm": f.ssm, "mask": f.mask,
                      "label": float(label), "source": source, "title": title, "wav": wav})

    # Stratified-ish split: hold out ~18% per class for validation.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(feats))
    val_ids, train_ids = set(), set()
    per_cls = {0: 0, 1: 0}
    n_val_each = max(3, int(0.18 * len(feats) / 2))
    for j in idx:
        lab = int(feats[j]["label"])
        if per_cls[lab] < n_val_each:
            val_ids.add(int(j)); per_cls[lab] += 1
        else:
            train_ids.add(int(j))
    train_items = [feats[j] for j in train_ids]
    val_items = [feats[j] for j in val_ids]

    example_meta = _save_examples(feats, examples_dir)

    model = FusionSegmentTransformer(
        embed_dim=EMBED_DIM, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=8,
        emb_stream_layers=3, ssm_stream_layers=2, dropout=0.1,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n_pos = sum(1 for f in train_items if f["label"] == 1.0)
    n_neg = len(train_items) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos else 1.0
    loss_fn = BCEWithLogits(pos_weight=pos_weight)

    best_auc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_items))
        for k in range(0, len(perm), 8):
            batch = [train_items[j] for j in perm[k:k + 8]]
            E, S, M, y = _collate(batch)
            E, S, M, y = E.to(dev), S.to(dev), M.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(E, ssm=S, key_padding_mask=M), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = compute_metrics(*_eval(model, val_items, dev))
        if val.auc == val.auc and val.auc >= best_auc:
            best_auc = val.auc
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        if epoch % 8 == 0 or epoch == epochs - 1:
            _log.info("epoch %d/%d val_auc=%.3f acc=%.3f", epoch + 1, epochs, val.auc, val.accuracy)

    model.load_state_dict(best_state)
    ckpt_path = out_dir / CKPT_NAME
    torch.save({"model_state": best_state, "embed_dim": EMBED_DIM, "seq_len": SEQ_LEN,
                "d_model": D_MODEL, "val_auc": best_auc}, ckpt_path)

    meta = {
        "val_auc": round(best_auc, 4),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_real": sum(1 for f in feats if f["source"] == "real"),
        "seconds": seconds,
        "examples": example_meta,
        "embed_dim": EMBED_DIM,
        "seq_len": SEQ_LEN,
    }
    meta_path = out_dir / META_NAME
    meta_path.write_text(json.dumps(meta, indent=2))
    _log.info("demo assets ready: ckpt=%s val_auc=%.3f (%d real clips)", ckpt_path, best_auc, meta["n_real"])
    return DemoBundle(ckpt_path, meta_path, examples_dir)


def _base_track(title: str) -> str:
    """Strip clip suffixes so distinct pieces are recognized (dedupe the gallery)."""
    return title.split(" · ")[0].replace(" (shuffled)", "").strip()


def _save_examples(feats: list[dict], examples_dir: Path) -> list[dict]:
    """Bundle a gallery: distinct real pieces (human + shuffled) plus one synth of each."""
    picks: list[dict] = []
    seen_tracks: set[str] = set()

    def take(source, label, limit, distinct=True):
        n = 0
        for f in feats:
            base = _base_track(f["title"])
            if f["source"] == source and int(f["label"]) == label and (not distinct or base not in seen_tracks):
                picks.append(f)
                if distinct:
                    seen_tracks.add(base)
                n += 1
                if n >= limit:
                    break

    take("real", 0, 4)      # real music, distinct pieces (structured)
    take("synth", 0, 1, distinct=False)   # synthetic loop (structured)
    take("synth", 1, 2, distinct=False)   # synthetic drift (unstructured)
    take("noise", 1, 1, distinct=False)   # unstructured noise

    example_meta = []
    for f in picks:
        label = "human" if int(f["label"]) == 0 else "ai_like"
        fname = f"{label}__{_slug(f['title'])}.wav"
        save_audio(examples_dir / fname, f["wav"], SR)
        example_meta.append({"file": fname, "title": f["title"], "source": f["source"],
                             "expected": "structured" if int(f["label"]) == 0 else "unstructured"})
    return example_meta


def _collate(batch):
    E = torch.stack([b["embeddings"] for b in batch])
    S = torch.stack([b["ssm"] for b in batch])
    M = torch.stack([b["mask"] for b in batch])
    y = torch.tensor([b["label"] for b in batch])
    return E, S, M, y


@torch.no_grad()
def _eval(model, items, dev):
    model.eval()
    E, S, M, y = _collate(items)
    logits = model(E.to(dev), ssm=S.to(dev), key_padding_mask=M.to(dev))
    return logits.cpu(), y


def load_demo_model(ckpt_path: str | Path, device: torch.device) -> FusionSegmentTransformer:
    state = torch.load(ckpt_path, map_location=device)
    model = FusionSegmentTransformer(
        embed_dim=state.get("embed_dim", EMBED_DIM), seq_len=state.get("seq_len", SEQ_LEN),
        d_model=state.get("d_model", D_MODEL), n_heads=8, emb_stream_layers=3, ssm_stream_layers=2,
    )
    model.load_state_dict(state["model_state"])
    return model.to(device).eval()
