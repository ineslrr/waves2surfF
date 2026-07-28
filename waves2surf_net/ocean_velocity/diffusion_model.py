# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""A compact conditional U-Net for 2-D EDM diffusion.

The noise-conditioning and adaptive residual blocks are a small, native
``[B,C,H,W]`` adaptation of ideas in oBottle/cBottle's SongUNet. Physical
time, temporal attention, calendar features, classifier heads, and flattened
space-time domains are intentionally omitted.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class FourierNoiseEmbedding(nn.Module):
    """Encode continuous log-noise levels with fixed Fourier frequencies."""

    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("Noise embedding requires at least two channels")
        half = channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.channels = channels

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        angles = noise.float().flatten()[:, None] * self.frequencies[None]
        result = torch.cat((angles.cos(), angles.sin()), dim=1)
        if result.shape[1] < self.channels:
            result = F.pad(result, (0, self.channels - result.shape[1]))
        return result


class ResidualBlock(nn.Module):
    """Residual convolution block adaptively scaled and shifted by noise."""

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.embedding(embedding).chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.conv2(F.silu(x))
        return (x + residual) / math.sqrt(2)


class ConditionalDiffusionUNet(nn.Module):
    """Denoise target fields while conditioning on aligned spatial inputs."""

    def __init__(
        self,
        target_channels: int,
        condition_channels: int,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        blocks_per_level: int = 1,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers cannot be empty")
        if blocks_per_level < 1:
            raise ValueError("blocks_per_level must be positive")
        self.target_channels = target_channels
        self.condition_channels = condition_channels
        embedding_dim = base_channels * 4
        self.noise_embedding = nn.Sequential(
            FourierNoiseEmbedding(base_channels),
            nn.Linear(base_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        widths = [base_channels * multiplier for multiplier in channel_multipliers]
        self.stem = nn.Conv2d(target_channels + condition_channels, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        current = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ResidualBlock(current, width, embedding_dim))
                current = width
            self.encoder.append(blocks)
            if level < len(widths) - 1:
                self.downsample.append(nn.Conv2d(current, widths[level + 1], 3, 2, 1))
                current = widths[level + 1]

        self.middle = nn.ModuleList(
            [ResidualBlock(current, current, embedding_dim) for _ in range(2)]
        )
        self.decoder = nn.ModuleList()
        for level in reversed(range(len(widths) - 1)):
            width = widths[level]
            blocks = nn.ModuleList(
                [ResidualBlock(current + width, width, embedding_dim)]
            )
            blocks.extend(
                ResidualBlock(width, width, embedding_dim)
                for _ in range(blocks_per_level - 1)
            )
            self.decoder.append(blocks)
            current = width

        self.out_norm = nn.GroupNorm(_groups(current), current)
        self.out = nn.Conv2d(current, target_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        noise_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.ndim != 4 or condition.ndim != 4:
            raise ValueError("noisy_target and condition must be [B,C,H,W]")
        if (
            noisy_target.shape[0] != condition.shape[0]
            or noisy_target.shape[-2:] != condition.shape[-2:]
        ):
            raise ValueError("target and condition batch/spatial shapes must match")
        if noisy_target.shape[1] != self.target_channels:
            raise ValueError(
                f"expected {self.target_channels} target channels, "
                f"received {noisy_target.shape[1]}"
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} condition channels, "
                f"received {condition.shape[1]}"
            )
        embedding = self.noise_embedding(noise_embedding)
        x = self.stem(torch.cat((noisy_target, condition), dim=1))
        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                x = block(x, embedding)
            skips.append(x)
            if level < len(self.downsample):
                x = self.downsample[level](x)
        for block in self.middle:
            x = block(x, embedding)
        for blocks, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
            for block in blocks:
                x = block(x, embedding)
        return self.out(F.silu(self.out_norm(x)))
