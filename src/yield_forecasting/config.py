from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration, resolving paths from the repository root."""
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "base.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][key] = str(
            candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        )
    return config
