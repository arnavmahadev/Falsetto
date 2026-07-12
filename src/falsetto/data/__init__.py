"""Data pipeline: audio loading, segmentation, augmentation, manifests, caching."""

from .audio import EXTRACTOR_AUDIO_SPEC, AudioSpec, conform, load_for_extractor, spec_for
from .augment import Augmentor
from .beat import BeatResult, BeatTracker, get_downbeats
from .cache import EmbeddingCache
from .segment_bars import BarSegmentation, four_bar_segments
from .segment_sequence import SegmentSequenceBuilder, pad_crop_sequence, stage1_embedding_fn
from .datasets import Stage1ClipDataset, collate_clips, make_dataloader
from .manifests import (
    MANIFEST_COLUMNS,
    SplitRatios,
    assign_splits,
    build_manifest,
    class_balance,
    load_manifest,
    save_manifest,
    scan_fakemusiccaps,
    stratified_group_split,
    verify_no_leakage,
)
from .segment_fixed import num_windows, segment_fixed

__all__ = [
    "AudioSpec",
    "EXTRACTOR_AUDIO_SPEC",
    "spec_for",
    "conform",
    "load_for_extractor",
    "segment_fixed",
    "num_windows",
    "Augmentor",
    "SplitRatios",
    "MANIFEST_COLUMNS",
    "build_manifest",
    "assign_splits",
    "stratified_group_split",
    "verify_no_leakage",
    "class_balance",
    "save_manifest",
    "load_manifest",
    "scan_fakemusiccaps",
    "Stage1ClipDataset",
    "collate_clips",
    "make_dataloader",
    "EmbeddingCache",
    "BeatTracker",
    "BeatResult",
    "get_downbeats",
    "four_bar_segments",
    "BarSegmentation",
    "SegmentSequenceBuilder",
    "pad_crop_sequence",
    "stage1_embedding_fn",
]
