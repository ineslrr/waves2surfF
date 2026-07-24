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


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON experiment file and validate its minimum data contract.

    The returned dictionary is passed unchanged to dataset, model, loss, and
    training factories.  It is also saved inside every checkpoint, which makes
    trained weights self-describing.
    """
    with open(path) as handle:
        config = json.load(handle)

    # These keys are the minimum information needed to construct a complete
    # experiment. Optional settings retain documented defaults elsewhere.
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
    # Validate metadata conditioning separately because the model input shape
    # depends on this choice.
    conditioning = config["model"].get("conditioning", "none")
    valid_conditioning = {"none", "extra_channels", "broadcast", "film"}
    if conditioning not in valid_conditioning:
        raise ValueError(
            f"model.conditioning must be one of {sorted(valid_conditioning)}"
        )
    # Calendar features are appended to file-based scalar metadata, so both
    # contribute one element to the metadata vector.
    metadata_dim = len(config["data"].get("metadata_variables", [])) + len(
        config["data"].get("calendar_features", [])
    )
    if conditioning != "none" and metadata_dim == 0:
        raise ValueError(
            f"model.conditioning={conditioning!r} requires metadata variables "
            "or calendar features"
        )
    return config
