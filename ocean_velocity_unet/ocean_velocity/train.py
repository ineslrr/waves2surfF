"""Training and evaluation for the generalized 2-D velocity model."""

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
from .model import UNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stats(config: dict[str, Any], name: str) -> ChannelStats | None:
    value = config["data"].get("normalization", {}).get(name)
    return ChannelStats.from_dict(value) if value else None


def make_dataset(config: dict[str, Any], split: str) -> NetCDFFieldDataset:
    data = config["data"]
    split_value = data["splits"][split]
    if isinstance(split_value, str):
        indices = np.load(split_value).tolist()
    else:
        indices = split_value
    return NetCDFFieldDataset(
        data["path"],
        data["input_variables"],
        data["target_variables"],
        indices,
        metadata_variables=data.get("metadata_variables", []),
        time_variable=data.get("time_variable", "time"),
        calendar_features=data.get("calendar_features", []),
        valid_mask_variable=data.get("valid_mask_variable"),
        spatial_slice=data.get("spatial_slice"),
        input_stats=_stats(config, "input"),
        metadata_stats=_stats(config, "metadata"),
        target_stats=_stats(config, "target"),
    )


def make_model(config: dict[str, Any]) -> UNet:
    data = config["data"]
    model = config["model"]
    metadata_dim = len(data.get("metadata_variables", [])) + len(
        data.get("calendar_features", [])
    )
    return UNet(
        len(data["input_variables"]),
        len(data["target_variables"]),
        model["base_channels"],
        model.get("normalization", "group"),
        metadata_dim=metadata_dim,
        conditioning=model.get("conditioning", "none"),
        metadata_channels=model.get("metadata_channels", 4),
        metadata_width=model.get("metadata_width", 32),
    )


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
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
    total = sum(weights.get(name, 0.0) * value for name, value in pieces.items())
    return total, {name: value.item() for name, value in pieces.items()}


def evaluate(model, loader, device, config) -> dict[str, Any]:
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
    training = config["training"]
    seed_everything(int(training.get("seed", 42)))
    device = torch.device(
        training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    train_dataset = make_dataset(config, "train")
    validation_dataset = make_dataset(config, "validation")
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
        initial = torch.load(
            training["initialize_from"], map_location=device, weights_only=False
        )
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"]
    )

    best_loss = float("inf")
    history = []
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, metadata)
            loss, _ = _loss(prediction, y, mask, config)
            loss.backward()
            if training.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                )
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        scheduler.step()

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
        with open(output_dir / "history.json", "w") as handle:
            json.dump(history, handle, indent=2)

    torch.save(
        {"model": model.state_dict(), "epoch": training["epochs"], "config": config},
        output_dir / "last.pt",
    )
