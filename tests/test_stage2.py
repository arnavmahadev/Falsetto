"""Phase 4 & 5 tests: SSM, 4-bar segmentation, Segment/Fusion transformers, training."""

from __future__ import annotations

import torch

from falsetto.config.schema import Config
from falsetto.data.segment_bars import four_bar_segments
from falsetto.data.segment_sequence import pad_crop_sequence
from falsetto.models.fusion import FusionSegmentTransformer
from falsetto.models.segment_transformer import SegmentTransformer
from falsetto.models.ssm import self_similarity_matrix
from falsetto.training.train_stage2 import train_stage2_from_sequences


# --------------------------------------------------------------------------- #
# T-27 SSM
# --------------------------------------------------------------------------- #
def test_ssm_properties():
    E = torch.randn(48, 32)
    ssm = self_similarity_matrix(E)
    assert ssm.shape == (48, 48)
    assert torch.allclose(ssm, ssm.T, atol=1e-5)  # symmetric
    assert torch.allclose(torch.diag(ssm), torch.ones(48), atol=1e-5)  # diag ~1
    assert (ssm >= 0).all() and (ssm <= 1.0 + 1e-5).all()


def test_ssm_batched_and_masked():
    E = torch.randn(2, 10, 16)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 7:] = True
    ssm = self_similarity_matrix(E, key_padding_mask=mask)
    assert ssm.shape == (2, 10, 10)
    # padded rows/cols zeroed
    assert ssm[0, 7:, :].abs().sum() == 0
    assert ssm[0, :, 7:].abs().sum() == 0


# --------------------------------------------------------------------------- #
# T-25 four-bar segmentation
# --------------------------------------------------------------------------- #
def test_four_bar_segmentation_downbeat():
    sr = 16000
    wave = torch.randn(1, sr * 20)  # 20 s
    downbeats = [float(i) for i in range(0, 20)]  # a downbeat every second
    seg = four_bar_segments(wave, sr, downbeats, bars_per_segment=4)
    assert seg.method == "downbeat"
    # boundaries every 4 downbeats -> ~5 segments over 20 s
    assert len(seg.segments) >= 4
    assert all(s.dim() == 2 for s in seg.segments)


def test_four_bar_segmentation_fallback():
    sr = 16000
    wave = torch.randn(1, sr * 20)
    seg = four_bar_segments(wave, sr, downbeats=[], fallback_clip_seconds=10.0)
    assert seg.method == "fixed_fallback"
    assert len(seg.segments) == 2  # 20 s / 10 s


def test_pad_crop_sequence():
    E = torch.randn(30, 64)
    out, mask = pad_crop_sequence(E, n=48)
    assert out.shape == (48, 64)
    assert mask[:30].sum() == 0 and mask[30:].all()  # first 30 real, rest padded
    # crop case
    out2, mask2 = pad_crop_sequence(torch.randn(60, 64), n=48)
    assert out2.shape == (48, 64) and not mask2.any()


# --------------------------------------------------------------------------- #
# T-28 Segment Transformer & T-34 Fusion
# --------------------------------------------------------------------------- #
def test_segment_transformer_forward():
    model = SegmentTransformer(embed_dim=768, seq_len=48, d_model=64, n_heads=4, n_layers=2)
    E = torch.randn(3, 48, 768)
    mask = torch.zeros(3, 48, dtype=torch.bool)
    mask[0, 40:] = True
    logits = model(E, key_padding_mask=mask)
    assert logits.shape == (3,)
    logits.sum().backward()


def test_fusion_segment_transformer_forward_and_gate():
    model = FusionSegmentTransformer(embed_dim=768, seq_len=48, d_model=64, n_heads=4)
    E = torch.randn(2, 48, 768)
    logits, g = model(E, return_gate=True)
    assert logits.shape == (2,)
    assert g.shape == (2, 48, 64)  # per-segment gate, accessible for viz
    assert (g >= 0).all() and (g <= 1).all()
    logits.sum().backward()


def test_fusion_plain_vs_gated():
    gated = FusionSegmentTransformer(embed_dim=32, seq_len=48, d_model=32, n_heads=4, fusion="gmu")
    plain = FusionSegmentTransformer(embed_dim=32, seq_len=48, d_model=32, n_heads=4, fusion="mean")
    assert gated.gate is not None
    assert plain.gate is None
    E = torch.randn(1, 48, 32)
    logits, g = plain(E, return_gate=True)
    assert g is None  # plain fusion has no gate


# --------------------------------------------------------------------------- #
# T-29 / T-35 Stage-2 training end-to-end (synthetic, separable sequences)
# --------------------------------------------------------------------------- #
def _make_sequences(n_per_class=24, seq_len=48, dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for label, offset in [(0, -1.0), (1, 1.0)]:
        for _ in range(n_per_class):
            E = torch.randn(seq_len, dim, generator=g) * 0.1 + offset  # class-separable mean
            mask = torch.zeros(seq_len, dtype=torch.bool)
            items.append({"embeddings": E, "mask": mask, "label": float(label)})
    return items


def test_stage2_fusion_training_beats_chance(tmp_path):
    train_items = _make_sequences(seed=1)
    val_items = _make_sequences(n_per_class=8, seed=2)

    cfg = Config(name="fst_e2e", seed=0, device="cpu", tracker="none")
    cfg.model.name = "fusion_segment_transformer"
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.emb_stream_layers = 2
    cfg.model.ssm_stream_layers = 1
    cfg.data.batch_size = 8
    cfg.train.epochs = 8
    cfg.train.lr = 1e-3
    cfg.train.amp = False
    cfg.train.early_stopping = False
    cfg.train.ckpt_dir = str(tmp_path / "ckpts")

    ckpt = train_stage2_from_sequences(cfg, train_items, val_items, embed_dim=64, seq_len=48)
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert state["metrics"]["auc"] > 0.75
