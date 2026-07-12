"""End-to-end inference (TASKS.md T-41).

    audio file -> resample -> beat-track -> 4-bar segments -> Stage-1 embeddings
              -> Stage-2 model -> P(AI) + label

:class:`Predictor` composes a trained Stage-1 detector (the segment embedder) and
a trained Stage-2 model (Segment or Fusion Transformer). ``from_configs`` builds
both from configs + checkpoints. For a Fusion model, the per-segment gate ``G``
can be returned for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config.schema import Config
from ..data.audio import load_for_extractor, spec_for
from ..data.beat import BeatTracker
from ..data.segment_sequence import SegmentSequenceBuilder, stage1_embedding_fn
from ..models.fusion import FusionSegmentTransformer
from ..utils.device import select_device
from ..utils.logging import get_logger

_log = get_logger("inference")


@dataclass
class Prediction:
    p_ai: float
    label: str  # "AI" | "Real"
    num_segments: int
    segmentation: str
    gate: torch.Tensor | None = None  # [N, d] for Fusion models


class Predictor:
    def __init__(
        self,
        stage1_model: torch.nn.Module,
        stage2_model: torch.nn.Module,
        extractor_name: str,
        seq_len: int = 48,
        device: torch.device | None = None,
        beat_tracker: BeatTracker | None = None,
    ) -> None:
        self.device = device or select_device("auto")
        self.extractor_name = extractor_name
        self.sample_rate = spec_for(extractor_name).sample_rate
        self.stage1 = stage1_model.to(self.device).eval()
        self.stage2 = stage2_model.to(self.device).eval()
        self.builder = SegmentSequenceBuilder(
            embed_fn=stage1_embedding_fn(self.stage1, self.device),
            sample_rate=self.sample_rate,
            beat_tracker=beat_tracker or BeatTracker(device=str(self.device)),
            max_segments=seq_len,
        )

    @classmethod
    def from_configs(
        cls,
        stage1_cfg: Config,
        stage1_ckpt: str | Path,
        stage2_cfg: Config,
        stage2_ckpt: str | Path,
        device: str = "auto",
    ) -> "Predictor":
        from ..models.stage1 import build_stage1_model
        from ..training.train_stage2 import build_stage2_model

        dev = select_device(device)
        stage1 = build_stage1_model(stage1_cfg)
        s1 = torch.load(stage1_ckpt, map_location=dev)
        stage1.load_state_dict(s1.get("model_state", s1))

        # Stage-2 consumes the Stage-1 head's Segment F.E Embedding (dim = d_model),
        # not the raw extractor feature dim.
        embed_dim = stage1_cfg.model.d_model
        stage2 = build_stage2_model(stage2_cfg.model, embed_dim, stage2_cfg.data.segment_length)
        s2 = torch.load(stage2_ckpt, map_location=dev)
        stage2.load_state_dict(s2.get("model_state", s2))

        return cls(stage1, stage2, stage1_cfg.extractor.name,
                   seq_len=stage2_cfg.data.segment_length, device=dev)

    @torch.no_grad()
    def predict_waveform(self, waveform: torch.Tensor, sr: int, return_gate: bool = False) -> Prediction:
        E, mask = self.builder.build(waveform, sr)
        E = E.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        gate = None
        if isinstance(self.stage2, FusionSegmentTransformer) and return_gate:
            logit, g = self.stage2(E, key_padding_mask=mask, return_gate=True)
            gate = g.squeeze(0).cpu()
        else:
            logit = self.stage2(E, key_padding_mask=mask)

        p_ai = torch.sigmoid(logit.reshape(-1)[0]).item()
        num_segments = int((~mask).sum().item())
        return Prediction(
            p_ai=p_ai,
            label="AI" if p_ai >= 0.5 else "Real",
            num_segments=num_segments,
            segmentation=self.builder.segmentation,
            gate=gate,
        )

    @torch.no_grad()
    def predict_file(self, path: str | Path, return_gate: bool = False) -> Prediction:
        waveform, sr = load_for_extractor(path, self.extractor_name)
        return self.predict_waveform(waveform, sr, return_gate=return_gate)
