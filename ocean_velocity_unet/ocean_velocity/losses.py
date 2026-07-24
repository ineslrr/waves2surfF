"""Masked pointwise, gradient, and spectral losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    mask = valid_mask.expand_as(prediction)
    error = torch.abs(prediction - target)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between log 2-D kinetic-energy spectra."""
    if prediction.shape[1] != 2:
        raise ValueError("Spectral kinetic-energy loss requires two output channels")
    height, width = prediction.shape[-2:]
    wy = torch.hann_window(height, device=prediction.device)
    wx = torch.hann_window(width, device=prediction.device)
    window = torch.outer(wy, wx)[None, None]
    mask = valid_mask.to(prediction.dtype)

    def energy(value: torch.Tensor) -> torch.Tensor:
        transform = torch.fft.rfft2(value * mask * window, dim=(-2, -1))
        return 0.5 * transform.abs().square().sum(dim=1)

    return F.mse_loss(torch.log(energy(prediction) + eps), torch.log(energy(target) + eps))


def velocity_gradients(
    velocity: torch.Tensor, dx: float = 1.0, dy: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, v = velocity[:, :1], velocity[:, 1:2]
    ux = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2 * dx)
    uy = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1]) / (2 * dy)
    vx = (v[..., 1:-1, 2:] - v[..., 1:-1, :-2]) / (2 * dx)
    vy = (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2 * dy)
    return ux, uy, vx, vy


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    pred = velocity_gradients(prediction, dx, dy)
    true = velocity_gradients(target, dx, dy)
    mask = valid_mask[..., 1:-1, 1:-1]
    pred_fields = (pred[2] - pred[1], pred[0] + pred[3])
    true_fields = (true[2] - true[1], true[0] + true[3])
    return sum(masked_l1(a, b, mask) for a, b in zip(pred_fields, true_fields))
