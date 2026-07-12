"""T-03: config loads into a typed object and round-trips to YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from falsetto.config import (
    Config,
    DataConfig,
    ExtractorConfig,
    from_dict,
    load_config,
    round_trip,
    save_config,
    to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_typed():
    cfg = Config()
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.extractor, ExtractorConfig)
    assert cfg.model.num_classes == 1


def test_round_trip_preserves_values():
    cfg = Config(name="exp", seed=7)
    cfg.data.batch_size = 16
    cfg.extractor.name = "wav2vec2"
    cfg.train.lr = 3e-4
    restored = round_trip(cfg)
    assert restored == cfg
    assert restored.data.batch_size == 16
    assert restored.extractor.name == "wav2vec2"
    assert restored.train.lr == 3e-4


def test_save_and_load(tmp_path):
    cfg = Config(name="io")
    cfg.model.d_model = 512
    path = save_config(cfg, tmp_path / "cfg.yaml")
    assert path.exists()
    loaded = load_config(path)
    assert loaded == cfg


def test_partial_yaml_merges_over_defaults():
    cfg = from_dict(Config, {"name": "partial", "train": {"epochs": 3}})
    assert cfg.name == "partial"
    assert cfg.train.epochs == 3
    # Untouched fields keep their defaults.
    assert cfg.train.lr == Config().train.lr
    assert cfg.extractor.name == Config().extractor.name


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        from_dict(Config, {"nope": 1})
    with pytest.raises(KeyError):
        from_dict(Config, {"train": {"not_a_field": 1}})


def test_example_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "stage1_mert_fakemusiccaps.yaml")
    assert cfg.name == "stage1_mert_fakemusiccaps"
    assert cfg.extractor.name == "mert"
    assert cfg.extractor.layer_strategy == "weighted"
    assert cfg.data.augment is True
    # Ensure round-trip on a real file too.
    assert round_trip(cfg) == cfg
    assert isinstance(to_dict(cfg), dict)
