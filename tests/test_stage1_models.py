"""Phase 3 tests: AudioCAT/FX-Segment shapes, losses, metrics, training loop."""

from __future__ import annotations

import numpy as np
import torch

from falsetto.models.audiocat import AudioCAT, WeightedLayerSum
from falsetto.models.fx_segment import FXSegment
from falsetto.training.losses import BCEWithLogits, FocalLoss, build_loss
from falsetto.training.metrics import compute_metrics


# --------------------------------------------------------------------------- #
# T-18 AudioCAT
# --------------------------------------------------------------------------- #
def test_audiocat_shapes_and_grad():
    model = AudioCAT(encoder_dim=768, d_model=64, n_heads=4, n_layers=2, num_latents=4)
    feats = torch.randn(3, 50, 768)  # [B, T, D_enc]
    logits = model(feats)
    assert logits.shape == (3,)  # one logit per clip
    logits.sum().backward()
    assert model.latents.grad is not None

    logits2, emb = model(feats, return_embedding=True)
    assert emb.shape == (3, 64)


def test_audiocat_respects_padding_mask():
    torch.manual_seed(0)
    model = AudioCAT(encoder_dim=32, d_model=32, n_heads=4, n_layers=2, num_latents=2).eval()
    feats = torch.randn(1, 10, 32)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 6:] = True  # pad last 4
    with torch.no_grad():
        out_masked = model(feats, key_padding_mask=mask)
        # Change padded region only; masked output should be identical.
        feats2 = feats.clone()
        feats2[0, 6:] = torch.randn(4, 32)
        out_masked2 = model(feats2, key_padding_mask=mask)
    assert torch.allclose(out_masked, out_masked2, atol=1e-5)


def test_weighted_layer_sum():
    wls = WeightedLayerSum(num_layers=13)
    layers = torch.randn(2, 13, 20, 768)  # [B, L, T, D]
    out = wls(layers)
    assert out.shape == (2, 20, 768)
    # Uniform init (zeros -> softmax uniform) == plain mean over layers
    assert torch.allclose(out, layers.mean(dim=1), atol=1e-5)


# --------------------------------------------------------------------------- #
# T-19 FX-Segment
# --------------------------------------------------------------------------- #
def test_fx_segment_shapes():
    model = FXSegment(encoder_dim=2048, d_model=64, n_heads=4, n_layers=2)
    feats = torch.randn(4, 16, 2048)
    logits = model(feats)
    assert logits.shape == (4,)
    logits.sum().backward()
    assert model.head.weight.grad is not None


# --------------------------------------------------------------------------- #
# T-20 losses
# --------------------------------------------------------------------------- #
def test_losses_finite_gradients():
    logits = torch.randn(8, requires_grad=True)
    targets = torch.randint(0, 2, (8,)).float()
    for loss_fn in (BCEWithLogits(), FocalLoss()):
        loss = loss_fn(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        logits.grad = None
    assert isinstance(build_loss("focal"), FocalLoss)


def test_focal_downweights_easy_examples():
    # A confident-correct example contributes less than an uncertain one.
    logits = torch.tensor([5.0, 0.1])
    targets = torch.tensor([1.0, 1.0])
    per = FocalLoss(reduction="none")(logits, targets)
    assert per[0] < per[1]


# --------------------------------------------------------------------------- #
# T-21 metrics
# --------------------------------------------------------------------------- #
def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, size=200)
    logits = rng.normal(size=200) + (targets - 0.5) * 2  # correlated with label
    res = compute_metrics(torch.tensor(logits), torch.tensor(targets))
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    assert res.accuracy == accuracy_score(targets, preds)
    assert abs(res.f1 - f1_score(targets, preds)) < 1e-9
    assert abs(res.auc - roc_auc_score(targets, probs)) < 1e-9


def test_metrics_perfect_separation():
    logits = torch.tensor([-5.0, -4.0, 4.0, 5.0])
    targets = torch.tensor([0, 0, 1, 1])
    res = compute_metrics(logits, targets)
    assert res.accuracy == 1.0
    assert res.auc == 1.0
    assert res.specificity == 1.0
