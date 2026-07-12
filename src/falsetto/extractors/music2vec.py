"""Music2Vec feature extractor (TASKS.md T-14).

`m-a-p/music2vec-v1` at 16 kHz (data2vec-style audio SSL, loaded via
``trust_remote_code``). Returns a ``[T, D]`` hidden-state sequence with the same
layer-strategy options as MERT (``last`` / ``mean`` / ``weighted``).
"""

from __future__ import annotations

from ..config.schema import ExtractorConfig
from .base import register_extractor
from .hf_ssl import HFSSLExtractor


@register_extractor("music2vec")
class Music2VecExtractor(HFSSLExtractor):
    def __init__(self, cfg: ExtractorConfig) -> None:
        super().__init__(
            cfg,
            default_pretrained="m-a-p/music2vec-v1",
            default_sample_rate=16000,
            default_embed_dim=768,
        )
