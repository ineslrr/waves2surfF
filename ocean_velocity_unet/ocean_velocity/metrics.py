"""Streaming masked regression metrics."""

from __future__ import annotations

import torch


class RegressionMetrics:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.reset()

    def reset(self) -> None:
        self.n = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_y = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_y2 = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_sq_error = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_abs_error = torch.zeros(self.channels, dtype=torch.float64)

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        mask = mask.expand_as(target).bool()
        for channel in range(self.channels):
            y = target[:, channel][mask[:, channel]].detach().double().cpu()
            p = prediction[:, channel][mask[:, channel]].detach().double().cpu()
            error = p - y
            self.n[channel] += y.numel()
            self.sum_y[channel] += y.sum()
            self.sum_y2[channel] += y.square().sum()
            self.sum_sq_error[channel] += error.square().sum()
            self.sum_abs_error[channel] += error.abs().sum()

    def compute(self) -> dict[str, list[float]]:
        n = self.n.clamp_min(1)
        total_variance = self.sum_y2 - self.sum_y.square() / n
        return {
            "rmse": torch.sqrt(self.sum_sq_error / n).tolist(),
            "mae": (self.sum_abs_error / n).tolist(),
            "r2": (1 - self.sum_sq_error / total_variance.clamp_min(1e-12)).tolist(),
        }
