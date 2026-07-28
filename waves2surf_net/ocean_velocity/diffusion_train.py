"""Small training helpers for notebook-driven conditional diffusion."""

from __future__ import annotations

import torch

from .diffusion import EDMPreconditioner, edm_loss


def diffusion_train_step(
    net: EDMPreconditioner,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    gradient_clip: float | None = 1.0,
) -> float:
    """Perform one optimizer update using a WaveCurrentDataset batch."""
    net.train()
    condition = batch["x"].to(device)
    target = batch["y"].to(device)
    mask = batch["valid_mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = edm_loss(net, target, condition, mask)
    loss.backward()
    if gradient_clip:
        torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
    optimizer.step()
    return float(loss.detach())


def train_diffusion_epoch(
    net: EDMPreconditioner,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    gradient_clip: float | None = 1.0,
    max_batches: int | None = None,
) -> float:
    """Train for one loader pass and return mean denoising loss."""
    total = 0.0
    count = 0
    for batch in loader:
        total += diffusion_train_step(
            net, batch, optimizer, device, gradient_clip=gradient_clip
        )
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    return total / max(count, 1)
