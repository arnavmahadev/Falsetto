"""Phase 7 tests: end-to-end Predictor (T-41) and the falsetto CLI predict (T-42)."""

from __future__ import annotations

import math

import torch

from falsetto.cli import main as cli_main
from falsetto.config import Config, save_config
from falsetto.config.schema import ExtractorConfig, ModelConfig
from falsetto.inference.predict import Predictor
from falsetto.models.fusion import FusionSegmentTransformer
from falsetto.models.segment_transformer import SegmentTransformer
from falsetto.models.stage1 import build_stage1_from_parts


def _track(sr=24000, seconds=8.0, freq=220.0):
    t = torch.arange(int(sr * seconds)) / sr
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


def _stage1(d_model=32):
    ext = ExtractorConfig(name="dummy", embed_dim=64, sample_rate=24000)
    mdl = ModelConfig(name="audiocat", d_model=d_model, n_heads=4, n_layers=1, num_latents=2)
    return build_stage1_from_parts(ext, mdl), d_model


def test_predictor_segment_transformer():
    stage1, d_model = _stage1()
    stage2 = SegmentTransformer(embed_dim=d_model, seq_len=48, d_model=32, n_heads=4, n_layers=1)
    predictor = Predictor(stage1, stage2, extractor_name="dummy", seq_len=48, device=torch.device("cpu"))
    pred = predictor.predict_waveform(_track(), 24000)
    assert 0.0 <= pred.p_ai <= 1.0
    assert pred.label in ("AI", "Real")
    assert pred.num_segments > 0


def test_predictor_fusion_returns_gate():
    stage1, d_model = _stage1()
    stage2 = FusionSegmentTransformer(embed_dim=d_model, seq_len=48, d_model=32, n_heads=4)
    predictor = Predictor(stage1, stage2, extractor_name="dummy", seq_len=48, device=torch.device("cpu"))
    pred = predictor.predict_waveform(_track(), 24000, return_gate=True)
    assert pred.gate is not None
    assert pred.gate.shape[0] == 48  # [N, d]


def test_cli_predict_end_to_end(tmp_path, capsys):
    from falsetto.data.audio import spec_for  # noqa: F401
    from falsetto.utils.audio_io import save_audio

    # Build + save Stage-1 (dummy+audiocat) and Stage-2 (segment_transformer) checkpoints.
    stage1, d_model = _stage1(d_model=32)
    stage2 = SegmentTransformer(embed_dim=d_model, seq_len=48, d_model=32, n_heads=4, n_layers=1)

    s1_ckpt = tmp_path / "s1.pt"
    s2_ckpt = tmp_path / "s2.pt"
    torch.save({"model_state": stage1.state_dict()}, s1_ckpt)
    torch.save({"model_state": stage2.state_dict()}, s2_ckpt)

    cfg1 = Config(name="s1")
    cfg1.extractor = ExtractorConfig(name="dummy", embed_dim=64, sample_rate=24000)
    cfg1.model = ModelConfig(name="audiocat", d_model=32, n_heads=4, n_layers=1, num_latents=2)
    cfg2 = Config(name="s2")
    cfg2.model = ModelConfig(name="segment_transformer", d_model=32, n_heads=4, n_layers=1)
    cfg2.data.segment_length = 48
    c1 = save_config(cfg1, tmp_path / "c1.yaml")
    c2 = save_config(cfg2, tmp_path / "c2.yaml")

    wav = tmp_path / "song.wav"
    save_audio(wav, _track(), 24000)

    rc = cli_main([
        "predict", str(wav),
        "--stage1-config", str(c1), "--stage1-ckpt", str(s1_ckpt),
        "--stage2-config", str(c2), "--stage2-ckpt", str(s2_ckpt),
        "--device", "cpu",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "P(AI)" in out
