"""Models: Stage-1 heads, Stage-2 Segment/Fusion Transformers, SSM."""

from .audiocat import AudioCAT, WeightedLayerSum
from .fusion import FusionSegmentTransformer
from .fx_segment import FXSegment
from .segment_transformer import SegmentTransformer
from .ssm import self_similarity_matrix
from .stage1 import (
    Stage1Detector,
    build_head,
    build_stage1_from_parts,
    build_stage1_model,
)

__all__ = [
    "AudioCAT",
    "WeightedLayerSum",
    "FXSegment",
    "Stage1Detector",
    "build_head",
    "build_stage1_model",
    "build_stage1_from_parts",
    "self_similarity_matrix",
    "SegmentTransformer",
    "FusionSegmentTransformer",
]
