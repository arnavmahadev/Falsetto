"""Shared HuggingFace self-supervised extractor base.

MERT, Music2Vec and Wav2Vec 2.0 all follow the same shape: a feature-extractor
"processor" normalizes the waveform, a transformer encoder produces per-frame
hidden states, and we reduce the layer axis by a configurable strategy:

    "last"     -> the final layer            [T, D]
    "mean"     -> mean over all layers       [T, D]
    "weighted" -> all layers stacked         [L, T, D]  (for a learnable sum
                  downstream); :meth:`extract` falls back to the layer mean so it
                  still returns [T, D], while :meth:`extract_layers` exposes the
                  full stack for a model-side WeightedLayerSum.
"""

from __future__ import annotations

import torch

from ..config.schema import ExtractorConfig
from .base import FeatureExtractor


class HFSSLExtractor(FeatureExtractor):
    """Wrap a HuggingFace audio SSL model as a :class:`FeatureExtractor`."""

    def __init__(
        self,
        cfg: ExtractorConfig,
        default_pretrained: str,
        default_sample_rate: int,
        default_embed_dim: int,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.pretrained = cfg.pretrained or default_pretrained
        self.sample_rate = cfg.sample_rate or default_sample_rate
        self.layer_strategy = cfg.layer_strategy
        self.returns_sequence = True

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.pretrained, trust_remote_code=cfg.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.pretrained, trust_remote_code=cfg.trust_remote_code
        )
        self.model.eval()
        # Prefer the model's reported hidden size over the config default.
        self.embed_dim = int(getattr(self.model.config, "hidden_size", default_embed_dim))
        self.num_layers = int(getattr(self.model.config, "num_hidden_layers", 12)) + 1

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _hidden_states(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return all layers stacked as ``[L, T, D]`` for one clip."""
        x = self._as_mono_1d(waveform).to(torch.float32).cpu().numpy()
        inputs = self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            out = self.model(input_values, output_hidden_states=True)
        # tuple of [B=1, T, D] -> [L, T, D]
        return torch.stack(out.hidden_states, dim=0).squeeze(1)

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        layers = self._hidden_states(waveform)  # [L, T, D]
        if self.layer_strategy == "last":
            return layers[-1]
        # "mean" and "weighted" both reduce to the layer mean here; the learnable
        # weighted sum consumes extract_layers() instead.
        return layers.mean(dim=0)

    @torch.no_grad()
    def extract_layers(self, waveform: torch.Tensor) -> torch.Tensor:
        """All hidden-state layers ``[L, T, D]`` (for a learnable WeightedLayerSum)."""
        return self._hidden_states(waveform)
