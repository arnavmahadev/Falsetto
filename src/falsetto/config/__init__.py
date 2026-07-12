"""Configuration system: typed dataclasses + YAML round-trip."""

from .io import from_dict, load_config, round_trip, save_config, to_dict
from .schema import (
    Config,
    DataConfig,
    EvalConfig,
    ExtractorConfig,
    ModelConfig,
    TrainConfig,
)

__all__ = [
    "Config",
    "DataConfig",
    "ExtractorConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "load_config",
    "save_config",
    "to_dict",
    "from_dict",
    "round_trip",
]
