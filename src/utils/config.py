"""YAML configuration loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
