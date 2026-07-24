"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        config = json.load(handle)
    required = {
        "data": ["path", "input_variables", "target_variables", "splits"],
        "model": ["base_channels"],
        "training": ["epochs", "batch_size", "learning_rate"],
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"Missing configuration section {section!r}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key {section}.{key}")
    conditioning = config["model"].get("conditioning", "none")
    valid_conditioning = {"none", "extra_channels", "broadcast", "film"}
    if conditioning not in valid_conditioning:
        raise ValueError(
            f"model.conditioning must be one of {sorted(valid_conditioning)}"
        )
    metadata_dim = len(config["data"].get("metadata_variables", [])) + len(
        config["data"].get("calendar_features", [])
    )
    if conditioning != "none" and metadata_dim == 0:
        raise ValueError(
            f"model.conditioning={conditioning!r} requires metadata variables "
            "or calendar features"
        )
    return config
