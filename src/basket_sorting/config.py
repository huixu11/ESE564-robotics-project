from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and return a mutable dictionary."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if config_path != DEFAULT_CONFIG_PATH:
        with config_path.open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        config = _deep_update(deepcopy(config), override)
    return config


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
