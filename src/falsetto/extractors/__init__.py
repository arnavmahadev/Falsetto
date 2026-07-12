"""Stage-1 feature extractors + common interface.

Importing this package registers every extractor with the registry so
:func:`build_extractor` can resolve it by name.
"""

from .base import (
    DummyExtractor,
    FeatureExtractor,
    available_extractors,
    build_extractor,
    register_extractor,
)
from .fxencoder import FXencoderExtractor
from .mert import MERTExtractor
from .muffin import MuffinExtractor
from .music2vec import Music2VecExtractor
from .wav2vec2 import Wav2Vec2Extractor

__all__ = [
    "FeatureExtractor",
    "DummyExtractor",
    "build_extractor",
    "register_extractor",
    "available_extractors",
    "MERTExtractor",
    "Wav2Vec2Extractor",
    "Music2VecExtractor",
    "FXencoderExtractor",
    "MuffinExtractor",
]
