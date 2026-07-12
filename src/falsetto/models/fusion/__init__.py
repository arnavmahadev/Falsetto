"""Fusion Segment Transformer (Paper 2) — streams, cross-attention, gate."""

from .cross_attention import BiDirectionalCrossAttention
from .embedding_stream import EmbeddingStream
from .fusion_segment_transformer import FusionSegmentTransformer
from .gate import GatedMultimodalUnit
from .ssm_stream import SSMStream

__all__ = [
    "EmbeddingStream",
    "SSMStream",
    "BiDirectionalCrossAttention",
    "GatedMultimodalUnit",
    "FusionSegmentTransformer",
]
