"""Wav2Vec 2.0 feature extractor (TASKS.md T-13).

`facebook/wav2vec2-base` at 16 kHz.

Dim note (documented choice): the paper cites 512, but HF's ``last_hidden_state``
(the transformer output) is **768**-dim while the CNN feature-encoder output is
512-dim. This wrapper returns the **768-dim transformer sequence** by default
(``feature_encoder=False``); set ``feature_encoder=True`` (via config
``layer_strategy="feature_encoder"``) to return the **512-dim** conv output
instead, matching the paper's cited dimensionality.
"""

from __future__ import annotations

import torch

from ..config.schema import ExtractorConfig
from .base import register_extractor
from .hf_ssl import HFSSLExtractor


@register_extractor("wav2vec2")
class Wav2Vec2Extractor(HFSSLExtractor):
    def __init__(self, cfg: ExtractorConfig) -> None:
        self._feature_encoder = cfg.layer_strategy == "feature_encoder"
        super().__init__(
            cfg,
            default_pretrained="facebook/wav2vec2-base",
            default_sample_rate=16000,
            default_embed_dim=768,
        )
        if self._feature_encoder:
            self.embed_dim = 512  # CNN feature-encoder output dim

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._feature_encoder:
            return super().extract(waveform)
        # 512-dim conv feature-encoder output: [D, T'] -> [T', D]
        x = self._as_mono_1d(waveform).to(torch.float32).cpu().numpy()
        inputs = self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        feats = self.model.feature_extractor(input_values)  # [B, 512, T']
        return feats.squeeze(0).transpose(0, 1).contiguous()
