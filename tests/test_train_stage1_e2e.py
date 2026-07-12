"""End-to-end Stage-1 training smoke test (T-22 done-when: beats chance + ckpt saved).

Uses the network-free dummy extractor on a synthetic, trivially-separable dataset
(loud tone = ai, near-silent = real) to exercise the whole loop: dataset ->
dataloader -> extractor -> AudioCAT -> loss -> optimizer -> early stop -> metrics
-> checkpoint. It asserts the model learns (val AUC well above chance).
"""

from __future__ import annotations

import math

import torch

from falsetto.config.schema import Config
from falsetto.data.manifests import assign_splits, build_manifest, save_manifest
from falsetto.training.train_stage1 import train_stage1_from_config
from falsetto.utils.audio_io import save_audio


def _make_dataset(root, sr=24000, n_per_class=30):
    records = []
    t = torch.arange(sr) / sr
    tone = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
    for i in range(n_per_class):
        # ai = loud tone, real = near silence -> separable after dummy projection
        ai = tone * 1.0
        real = tone * 0.01
        save_audio(root / "ai" / f"ai{i:03d}.wav", ai, sr)
        save_audio(root / "real" / f"real{i:03d}.wav", real, sr)
        records.append({"filepath": str(root / "ai" / f"ai{i:03d}.wav"),
                        "track_id": f"ai{i}", "label": 1, "source": "synth"})
        records.append({"filepath": str(root / "real" / f"real{i:03d}.wav"),
                        "track_id": f"real{i}", "label": 0, "source": "real"})
    df = build_manifest(records, dataset="synth")
    return assign_splits(df, seed=0)


def test_stage1_training_beats_chance(tmp_path):
    manifest = _make_dataset(tmp_path / "audio")
    save_manifest(manifest, tmp_path / "manifest.csv")

    cfg = Config(name="e2e", seed=0, device="cpu", tracker="none", output_dir=str(tmp_path / "runs"))
    cfg.extractor.name = "dummy"
    cfg.extractor.embed_dim = 256
    cfg.extractor.sample_rate = 24000
    cfg.data.manifest = str(tmp_path / "manifest.csv")
    cfg.data.clip_seconds = 1.0
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    cfg.data.augment = False
    cfg.model.name = "audiocat"
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.n_layers = 2
    cfg.model.num_latents = 2
    cfg.train.epochs = 10
    cfg.train.lr = 1e-3
    cfg.train.amp = False
    cfg.train.early_stopping = False
    cfg.train.ckpt_dir = str(tmp_path / "ckpts")

    ckpt_path = train_stage1_from_config(cfg, manifest=manifest)

    assert ckpt_path.exists()
    state = torch.load(ckpt_path, map_location="cpu")
    assert "model_state" in state
    # Best-checkpoint val metric should clearly beat chance (0.5).
    assert state["metrics"]["auc"] > 0.75
