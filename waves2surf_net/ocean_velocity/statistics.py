"""Estimate training-set normalization statistics in a streaming pass.

This is normally run once before training. The resulting means/stds are copied
into the experiment configuration so training, validation, test, and later
inference all use exactly the same transformation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .train import make_dataset


class Moments:
    """Accumulate count, sum, and squared sum for each channel."""

    def __init__(self, channels: int) -> None:
        """Allocate small float64 accumulators independent of dataset size."""
        self.count = torch.zeros(channels, dtype=torch.float64)
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.total_square = torch.zeros(channels, dtype=torch.float64)

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """Add valid values from one mini-batch to the running moments."""
        mask = mask.expand_as(values)
        for channel in range(values.shape[1]):
            selected = values[:, channel][mask[:, channel]].double()
            self.count[channel] += selected.numel()
            self.total[channel] += selected.sum()
            self.total_square[channel] += selected.square().sum()

    def result(self) -> dict[str, list[float]]:
        """Convert raw moments to channel means and population stds."""
        count = self.count.clamp_min(1)
        mean = self.total / count
        variance = self.total_square / count - mean.square()
        return {
            "mean": mean.tolist(),
            "std": variance.clamp_min(1e-16).sqrt().tolist(),
        }


def calculate_statistics(
    config: dict[str, Any], output_path: str | Path, batch_size: int = 4
) -> dict[str, Any]:
    """Compute input, metadata, and target statistics on the training split.

    Existing normalization is deliberately removed from a deep copy of the
    configuration: statistics must describe the original physical values, not
    values that have already been standardized.
    """
    # JSON round-tripping is a simple deep copy for this JSON-native config.
    raw_config = json.loads(json.dumps(config))
    raw_config["data"].pop("normalization", None)
    dataset = make_dataset(raw_config, "train")
    # A single worker avoids unnecessary file-process complexity for this
    # one-time sequential pass.
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    input_moments = Moments(
        len(config["data"]["input_variables"])
    )
    metadata_moments = Moments(
        len(config["data"].get("metadata_variables", []))
        + len(config["data"].get("calendar_features", []))
    )
    target_moments = Moments(len(config["data"]["target_variables"]))
    for batch in loader:
        # Spatial inputs and targets use the wet/finite pixel mask. Metadata has
        # one value per sample, so it uses the sample-level validity flag.
        input_moments.update(batch["x"], batch["valid_mask"])
        if batch["metadata"].shape[1]:
            metadata_moments.update(
                batch["metadata"][:, :, None, None],
                batch["metadata_valid"][:, None, None, None],
            )
        target_moments.update(batch["y"], batch["valid_mask"])
    result = {
        "input": input_moments.result(),
        "metadata": metadata_moments.result(),
        "target": target_moments.result(),
    }
    # JSON output can be inspected, versioned, and pasted directly into config.
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    return result
