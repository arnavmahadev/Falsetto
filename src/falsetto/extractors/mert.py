"""MERT feature extractor (TASKS.md T-12).

`m-a-p/MERT-v1-95M` at 24 kHz, loaded via ``transformers`` with
``trust_remote_code=True``. Returns the hidden-state **sequence** ``[T, 768]``
that AudioCAT consumes as key/value.

Layer strategy (configurable): ``last`` uses the final layer; ``mean`` averages
MERT's 13 layers; ``weighted`` exposes all layers via ``extract_layers`` for a
downstream learnable weighted sum (the documented choice for the headline runs).
"""

from __future__ import annotations

from ..config.schema import ExtractorConfig
from .base import register_extractor
from .hf_ssl import HFSSLExtractor


@register_extractor("mert")
class MERTExtractor(HFSSLExtractor):
    def __init__(self, cfg: ExtractorConfig) -> None:
        super().__init__(
            cfg,
            default_pretrained="m-a-p/MERT-v1-95M",
            default_sample_rate=24000,
            default_embed_dim=768,
        )
