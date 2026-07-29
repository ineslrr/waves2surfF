"""Assemble datasets/model/losses and run optimization or evaluation.

This module is the pipeline coordinator. Specialized details remain in
``data.py``, ``model.py``, ``losses.py``, and ``metrics.py``; the functions
here connect those pieces using one experiment configuration.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ChannelStats, NetCDFFieldDataset
from .losses import gradient_loss, masked_l1, spectral_loss
from .metrics import RegressionMetrics
from .model import Waves2SurfNet
from .patches import apply_patches
from .wavecurrentdataset import WaveCurrentDataset


def seed_everything(seed: int) -> None:
    """Seed common random-number generators for more reproducible comparisons.

    Exact bitwise reproducibility can still depend on hardware and PyTorch
    kernels, but fixed seeds keep data shuffling and initialization controlled
    across most runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stats(config: dict[str, Any], name: str) -> ChannelStats | None:
    """Extract one optional normalization group from the data configuration."""
    value = config["data"].get("normalization", {}).get(name)
    return ChannelStats.from_dict(value) if value else None


def make_dataset(
    config: dict[str, Any], split: str
):
    """Build a dataset for ``train``, ``validation``, or ``test``.

    Dual-source configs (``data.sources``) use :class:`WaveCurrentDataset`
    with date-range splits. Single-file configs (``data.path``) keep the
    original :class:`NetCDFFieldDataset` and index-list / ``.npy`` splits.

    Optional ``data.patches`` wraps the result in
    :class:`~ocean_velocity.patches.SpatialPatchDataset`.
    """
    data = config["data"]
    input_stats = _stats(config, "input")
    metadata_stats = _stats(config, "metadata")
    target_stats = _stats(config, "target")

    if "sources" in data:
        dataset = WaveCurrentDataset.from_config(
            config,
            split,
            input_stats=input_stats,
            metadata_stats=metadata_stats,
            target_stats=target_stats,
        )
    else:
        split_value = data["splits"][split]
        if isinstance(split_value, str):
            indices = np.load(split_value).tolist()
        else:
            indices = split_value
        dataset = NetCDFFieldDataset(
            data["path"],
            data["input_variables"],
            data["target_variables"],
            indices,
            metadata_variables=data.get("metadata_variables", []),
            time_variable=data.get("time_variable", "time"),
            calendar_features=data.get("calendar_features", []),
            valid_mask_variable=data.get("valid_mask_variable"),
            spatial_slice=data.get("spatial_slice"),
            input_stats=input_stats,
            metadata_stats=metadata_stats,
            target_stats=target_stats,
        )
    return apply_patches(dataset, config, split)


def make_model(config: dict[str, Any]) -> Waves2SurfNet:
    """Construct Waves2SurfNet with channels matching configured variables."""
    data = config["data"]
    model = config["model"]
    # File-based metadata and derived calendar values are concatenated by the
    # dataset into one vector in this exact order.
    metadata_dim = len(data.get("metadata_variables", [])) + len(
        data.get("calendar_features", [])
    )
    return Waves2SurfNet(
        len(data["input_variables"]),
        len(data["target_variables"]),
        model["base_channels"],
        model.get("normalization", "group"),
        metadata_dim=metadata_dim,
        conditioning=model.get("conditioning", "none"),
        metadata_channels=model.get("metadata_channels", 4),
        metadata_width=model.get("metadata_width", 32),
        depth=int(model.get("depth", 4)),
    )


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the weighted differentiable objective for one mini-batch.

    Pointwise L1 is always calculated. More expensive scientific auxiliaries
    are calculated only when assigned nonzero weights.
    """
    weights = config["training"].get("loss_weights", {"pointwise": 1.0})
    pieces = {"pointwise": masked_l1(prediction, target, mask)}
    if weights.get("spectral", 0):
        pieces["spectral"] = spectral_loss(prediction, target, mask)
    if weights.get("gradient", 0):
        pieces["gradient"] = gradient_loss(
            prediction,
            target,
            mask,
            config["data"].get("dx", 1.0),
            config["data"].get("dy", 1.0),
        )
    # ``total`` drives backpropagation; detached scalar pieces are useful for
    # human-readable logging and debugging.
    total = sum(weights.get(name, 0.0) * value for name, value in pieces.items())
    return total, {name: value.item() for name, value in pieces.items()}


def evaluate(model, loader, device, config) -> dict[str, Any]:
    """Evaluate one complete split without updating model parameters.

    Loss remains in normalized training space for checkpoint selection.
    RMSE/MAE/R² are converted back to physical target units so reported values
    are scientifically interpretable.
    """
    # eval() changes layers such as BatchNorm; no_grad() below also avoids
    # storing activations needed only for backpropagation.
    model.eval()
    metrics = RegressionMetrics(len(config["data"]["target_variables"]))
    loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            metadata = batch["metadata"].to(device)
            mask = batch["valid_mask"].to(device)
            prediction = model(x, metadata)
            loss, _ = _loss(prediction, y, mask, config)
            loss_sum += loss.item()
            batches += 1
            # Undo target standardization for user-facing metrics. This affine
            # transformation is applied equally to prediction and truth.
            target_stats = _stats(config, "target")
            if target_stats:
                mean = torch.tensor(
                    target_stats.mean, device=device, dtype=y.dtype
                )[None, :, None, None]
                std = torch.tensor(
                    target_stats.std, device=device, dtype=y.dtype
                )[None, :, None, None]
                metrics.update(prediction * std + mean, y * std + mean, mask)
            else:
                metrics.update(prediction, y, mask)
    result = metrics.compute()
    result["loss"] = loss_sum / max(batches, 1)
    return result


def run_training(config: dict[str, Any], output_dir: str | Path) -> None:
    """Run the complete train/validate/checkpoint loop.

    Each epoch performs gradient-based updates on the training split, evaluates
    the untouched validation split, and saves the best validation checkpoint.
    The test split is intentionally absent here and is evaluated separately
    with ``evaluate.py`` after model selection.
    """
    training = config["training"]
    seed_everything(int(training.get("seed", 42)))
    device = torch.device(
        training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    # Every experiment gets a self-contained output directory. Saving the
    # resolved config first records the intended setup even if a job stops.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    train_dataset = make_dataset(config, "train")
    validation_dataset = make_dataset(config, "validation")
    # Pinned host memory and non-blocking transfers improve GPU input throughput.
    # Persistent workers avoid reopening NetCDF files every epoch.
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training.get("num_workers", 4),
        "pin_memory": device.type == "cuda",
        "persistent_workers": training.get("num_workers", 4) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = make_model(config).to(device)
    if training.get("initialize_from"):
        # Weight-only initialization supports a second loss stage while giving
        # the new objective a fresh optimizer and learning-rate schedule.
        initial = torch.load(
            training["initialize_from"], map_location=device, weights_only=False
        )
        model.load_state_dict(initial["model"])
    # AdamW is a robust default for convolutional networks; weight decay mildly
    # regularizes parameters independently of the adaptive gradient update.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 1e-5),
    )
    # Smoothly lower the learning rate so late epochs make finer updates.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"]
    )

    best_loss = float("inf")
    history = []
    for epoch in range(1, training["epochs"] + 1):
        # train() enables training-time layer behavior and gradient tracking.
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            # Standard PyTorch update: clear old gradients, predict, measure
            # error, backpropagate derivatives, then change parameters.
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, metadata)
            loss, _ = _loss(prediction, y, mask, config)
            loss.backward()
            if training.get("gradient_clip"):
                # Clipping guards against occasional unstable, very large
                # updates, especially when auxiliary losses are introduced.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                )
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        scheduler.step()

        # Validation happens after the epoch and never calls optimizer.step().
        validation = evaluate(model, validation_loader, device, config)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "validation": validation,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record))
        if validation["loss"] < best_loss:
            # Selection by validation—not test—loss avoids optimistic test
            # estimates. Save optimizer state so this checkpoint can be resumed.
            best_loss = validation["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_loss": best_loss,
                    "config": config,
                },
                output_dir / "best.pt",
            )
        # Rewrite a small history file each epoch so external jobs can monitor
        # progress and interrupted runs retain all completed records.
        with open(output_dir / "history.json", "w") as handle:
            json.dump(history, handle, indent=2)

    # ``last.pt`` is useful for diagnostics even when an earlier epoch was best.
    torch.save(
        {"model": model.state_dict(), "epoch": training["epochs"], "config": config},
        output_dir / "last.pt",
    )
