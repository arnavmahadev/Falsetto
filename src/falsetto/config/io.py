"""Config (de)serialization: dataclass <-> dict <-> YAML with round-trip fidelity.

``from_dict`` walks the dataclass field types so nested sections (``data``,
``extractor``, ...) rebuild as their proper dataclass, and unknown keys raise
instead of silently vanishing. ``load_config`` merges a partial YAML over the
defaults, so a config file only needs to state what it changes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from .schema import Config

T = TypeVar("T")


def to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a (possibly nested) dataclass into a plain dict."""
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"expected a dataclass instance, got {type(cfg)!r}")
    return dataclasses.asdict(cfg)


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass of type ``cls`` from a nested dict.

    Recurses into fields that are themselves dataclasses. Keys not present in
    the schema raise ``KeyError`` so typos in a YAML config fail loudly rather
    than being dropped.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data)!r}")

    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise KeyError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"valid keys: {sorted(field_names)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = hints[name]
        if _is_dataclass_type(field_type) and isinstance(value, dict):
            kwargs[name] = from_dict(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def save_config(cfg: Config, path: str | Path) -> Path:
    """Write ``cfg`` to a YAML file (creating parent dirs). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, default_flow_style=False)
    return path


def load_config(path: str | Path) -> Config:
    """Load a YAML config, merging it over :class:`Config` defaults."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(Config, raw)


def round_trip(cfg: Config) -> Config:
    """Convenience for tests: ``cfg -> yaml-safe dict -> cfg``."""
    return from_dict(Config, yaml.safe_load(yaml.safe_dump(to_dict(cfg))))
