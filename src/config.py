"""Tiny YAML config loader with ``${ENV_VAR}`` interpolation and caching.

All modules read configuration through here so the set of active data layers and
feature groups is controlled entirely by the files in ``configs/``.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import paths

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` tokens in strings with environment values.

    Unset variables resolve to an empty string (callers treat that as "missing").
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    """Load ``configs/<name>`` (``.yaml`` optional) with env interpolation.

    Returns an empty dict if the file is missing, so optional configs are safe.
    """
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path: Path = paths.CONFIGS / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _interpolate_env(data)


def data_sources() -> dict:
    """Return the ``data_sources.yaml`` mapping (source name -> settings)."""
    return load_yaml("data_sources")


def feature_groups() -> dict:
    """Return the enabled-flags mapping from ``feature_groups.yaml``."""
    return load_yaml("feature_groups").get("feature_groups", {})


def source_cfg(name: str) -> dict:
    """Config block for a single data source, e.g. ``source_cfg('odds')``."""
    return data_sources().get(name, {}) or {}


def is_enabled(source: str) -> bool:
    """True if a data source is switched on in ``data_sources.yaml``."""
    return bool(source_cfg(source).get("enabled", False))


def is_feature_group_enabled(group: str) -> bool:
    """True if a feature group is switched on in ``feature_groups.yaml``."""
    return bool(feature_groups().get(group, False))


def clear_cache() -> None:
    """Drop cached configs (used by tests that mutate config on disk)."""
    load_yaml.cache_clear()
