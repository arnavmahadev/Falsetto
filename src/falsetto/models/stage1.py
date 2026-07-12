"""Stage-1 detector: frozen extractor + attention head (TASKS.md T-18/T-19 glue).

:class:`Stage1Detector` runs a (frozen) Phase-2 extractor over a batch of clips,
pads the per-clip feature sequences to a common length with a key-padding mask,
and applies an AudioCAT or FX-Segment head to produce one logit per clip. The
extractor is always held in eval mode and run under ``no_grad`` so its BatchNorm
stats never update and no extractor grads are tracked.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import Config, ExtractorConfig, ModelConfig
from ..extractors.base import FeatureExtractor, build_extractor
from .audiocat import AudioCAT
from .fx_segment import FXSegment


class Stage1Detector(nn.Module):
    """Wrap ``extractor -> head`` into a waveform-in / logit-out module."""

    def __init__(self, extractor: FeatureExtractor, head: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor.freeze()
        self.head = head

    def train(self, mode: bool = True) -> "Stage1Detector":
        super().train(mode)
        self.extractor.eval()  # keep frozen extractor in eval regardless of parent mode
        return self

    @torch.no_grad()
    def _encode_batch(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the extractor over ``[B, C, S]`` -> padded ``([B, Tmax, D], mask [B, Tmax])``."""
        feats = [self.extractor.extract(waveforms[i]) for i in range(waveforms.size(0))]
        lengths = [f.size(0) for f in feats]
        t_max = max(lengths)
        dim = feats[0].size(1)
        device = feats[0].device
        batch = waveforms.new_zeros(len(feats), t_max, dim)
        mask = torch.ones(len(feats), t_max, dtype=torch.bool, device=device)
        for i, f in enumerate(feats):
            batch[i, : f.size(0)] = f
            mask[i, : f.size(0)] = False  # False = keep
        return batch, mask

    def forward(self, waveforms: torch.Tensor, return_embedding: bool = False):
        if waveforms.dim() == 2:  # [C, S] -> [1, C, S]
            waveforms = waveforms.unsqueeze(0)
        feats, mask = self._encode_batch(waveforms)
        return self.head(feats, key_padding_mask=mask, return_embedding=return_embedding)

    def forward_features(self, feats: torch.Tensor, key_padding_mask=None, return_embedding=False):
        """Head-only forward for precomputed/cached features."""
        return self.head(feats, key_padding_mask=key_padding_mask, return_embedding=return_embedding)


def build_head(model_cfg: ModelConfig, encoder_dim: int) -> nn.Module:
    """Instantiate the Stage-1 head named by ``model_cfg.name``."""
    name = model_cfg.name.lower()
    if name == "audiocat":
        return AudioCAT(
            encoder_dim=encoder_dim,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            num_latents=model_cfg.num_latents,
            num_classes=model_cfg.num_classes,
        )
    if name in ("fx_segment", "fxsegment", "fx-segment"):
        return FXSegment(
            encoder_dim=encoder_dim,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            num_classes=model_cfg.num_classes,
        )
    raise ValueError(f"unknown Stage-1 model {model_cfg.name!r}")


def build_stage1_model(cfg: Config) -> Stage1Detector:
    """Build ``extractor -> head`` from a full :class:`Config`."""
    extractor = build_extractor(cfg.extractor)
    head = build_head(cfg.model, encoder_dim=extractor.embed_dim)
    return Stage1Detector(extractor, head)


def build_stage1_from_parts(extractor_cfg: ExtractorConfig, model_cfg: ModelConfig) -> Stage1Detector:
    extractor = build_extractor(extractor_cfg)
    head = build_head(model_cfg, encoder_dim=extractor.embed_dim)
    return Stage1Detector(extractor, head)
