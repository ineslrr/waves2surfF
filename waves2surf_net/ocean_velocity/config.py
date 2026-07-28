"""Load the experiment configuration before data or GPUs are initialized.

Keeping scientific choices in JSON makes runs reproducible and lets the same
Python code move between laptops and HPC systems.  This module performs a few
early checks so a misspelled option fails immediately rather than halfway
through a long training job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _is_date_range(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "start" in value
        and "end" in value
        and isinstance(value["start"], str)
        and isinstance(value["end"], str)
    )


def _validate_splits(splits: Any, *, dual_source: bool) -> None:
    if not isinstance(splits, dict) or not splits:
        raise ValueError("data.splits must be a non-empty object")
    if dual_source:
        has_date = any(_is_date_range(v) for v in splits.values())
        if not has_date:
            raise ValueError(
                "dual-source configs require date-range splits, e.g. "
                '{"train": {"start": "20230101", "end": "20231231"}}'
            )
        if "train" not in splits:
            raise ValueError("data.splits must include train")
        if "evaluation" in splits:
            if not _is_date_range(splits["evaluation"]):
                raise ValueError("data.splits.evaluation must have start/end")
        else:
            for name in ("validation", "test"):
                if name not in splits or not _is_date_range(splits[name]):
                    raise ValueError(
                        "dual-source configs need validation/test date ranges "
                        "or a single evaluation range"
                    )
        return
    # Single-file mode: index lists, .npy paths, or date ranges (ignored here).
    if "train" not in splits:
        raise ValueError("data.splits must include train")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON experiment file and validate its minimum data contract.

    The returned dictionary is passed unchanged to dataset, model, loss, and
    training factories.  It is also saved inside every checkpoint, which makes
    trained weights self-describing.
    """
    with open(path) as handle:
        config = json.load(handle)

    if "data" not in config:
        raise ValueError("Missing configuration section 'data'")
    data = config["data"]
    dual_source = "sources" in data
    single_file = "path" in data
    if dual_source == single_file:
        raise ValueError(
            "data must define exactly one of 'path' (single NetCDF) or "
            "'sources' (wave/current daily folders)"
        )

    required_data = ["input_variables", "target_variables", "splits"]
    for key in required_data:
        if key not in data:
            raise ValueError(f"Missing configuration key data.{key}")

    if dual_source:
        sources = data["sources"]
        for name in ("waves", "currents"):
            if name not in sources:
                raise ValueError(f"Missing data.sources.{name}")
            if "dir" not in sources[name]:
                raise ValueError(f"Missing data.sources.{name}.dir")

    _validate_splits(data["splits"], dual_source=dual_source)

    required = {
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
    depth = config["model"].get("depth", 4)
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("model.depth must be an integer >= 1")
    metadata_dim = len(data.get("metadata_variables", [])) + len(
        data.get("calendar_features", [])
    )
    if conditioning != "none" and metadata_dim == 0:
        raise ValueError(
            f"model.conditioning={conditioning!r} requires metadata variables "
            "or calendar features"
        )
    if dual_source and data.get("metadata_variables"):
        raise ValueError(
            "data.metadata_variables is not supported with data.sources yet; "
            "use calendar_features only"
        )
    return config
