# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Elucidated Diffusion Model utilities for conditional field generation.

The preconditioning, weighted denoising objective, and second-order sampler
are adapted from oBottle/cBottle and the EDM reference implementation. They
are kept independent of the dataset and training loop for easy notebook use.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


class EDMPreconditioner(nn.Module):
    """Turn a noise-predicting U-Net into a clean-target denoiser."""

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> None:
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        self.model = model
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def forward(
        self, noisy_target: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        noisy_target = noisy_target.float()
        condition = torch.nan_to_num(condition.float())
        sigma = torch.as_tensor(
            sigma, device=noisy_target.device, dtype=torch.float32
        ).reshape(-1, 1, 1, 1)
        if sigma.shape[0] == 1 and noisy_target.shape[0] != 1:
            sigma = sigma.expand(noisy_target.shape[0], -1, -1, -1)
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / (sigma.square() + sigma_data**2).sqrt()
        c_in = 1 / (sigma_data**2 + sigma.square()).sqrt()
        c_noise = sigma.log().flatten() / 4
        residual = self.model(c_in * noisy_target, c_noise, condition)
        return c_skip * noisy_target + c_out * residual.float()

    @staticmethod
    def round_sigma(sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


def edm_loss(
    net: EDMPreconditioner,
    target: torch.Tensor,
    condition: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the EDM weighted MSE averaged over valid target elements."""
    sigma = (
        torch.randn(
            (target.shape[0], 1, 1, 1),
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        * p_std
        + p_mean
    ).exp()
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    denoised = net(target + noise * sigma, sigma, condition)
    weight = (sigma.square() + net.sigma_data**2) / (
        sigma * net.sigma_data
    ).square()
    error = weight * (denoised - target).square()
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(error.dtype).expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def edm_sample(
    net: EDMPreconditioner,
    condition: torch.Tensor,
    *,
    shape: tuple[int, int, int, int] | None = None,
    num_steps: int = 18,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    rho: float = 7.0,
    generator: torch.Generator | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Generate conditional samples with deterministic Heun integration."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least two")
    if shape is None:
        target_channels = getattr(net.model, "target_channels", None)
        if target_channels is None:
            raise ValueError("shape is required when model has no target_channels")
        shape = (
            condition.shape[0],
            target_channels,
            condition.shape[-2],
            condition.shape[-1],
        )
    if shape[0] != condition.shape[0] or shape[-2:] != condition.shape[-2:]:
        raise ValueError("sample shape and condition batch/spatial shapes must match")
    sigma_min = max(net.sigma_min, sigma_min or net.sigma_min)
    sigma_max = min(net.sigma_max, sigma_max or net.sigma_max)
    indices = torch.arange(num_steps, device=condition.device, dtype=torch.float64)
    steps = (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ).pow(rho)
    steps = torch.cat((steps, steps.new_zeros(1)))
    sample = torch.randn(
        shape, device=condition.device, dtype=condition.dtype, generator=generator
    ) * steps[0]
    for index, (current, following) in enumerate(zip(steps[:-1], steps[1:])):
        current_sample = sample
        denoised = net(current_sample, current, condition).to(torch.float64)
        derivative = (current_sample.to(torch.float64) - denoised) / current
        sample = current_sample + (following - current) * derivative
        if index < num_steps - 1:
            next_denoised = net(sample, following, condition).to(torch.float64)
            next_derivative = (sample - next_denoised) / following
            sample = current_sample + (following - current) * (
                derivative + next_derivative
            ) / 2
        if progress:
            progress(index + 1, num_steps)
    return sample.to(condition.dtype)


def estimate_sigma_data(
    target_std: list[float] | tuple[float, ...] | None = None,
) -> float:
    """Choose sigma_data from normalized target standard deviations.

    Standardized targets should use the conventional value 1.0.
    """
    if not target_std:
        return 1.0
    return math.sqrt(sum(value * value for value in target_std) / len(target_std))
