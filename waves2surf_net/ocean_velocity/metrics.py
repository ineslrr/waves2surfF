"""Accumulate evaluation metrics without retaining full prediction fields."""

from __future__ import annotations

import torch


class RegressionMetrics:
    """Streaming per-channel RMSE, MAE, and coefficient of determination.

    Validation/test datasets can be much larger than memory. This accumulator
    stores only sufficient statistics and is updated once per batch.
    """

    def __init__(self, channels: int) -> None:
        """Create an accumulator for the requested number of target channels."""
        self.channels = channels
        self.reset()

    def reset(self) -> None:
        """Clear all running totals so the object can be reused."""
        # float64 accumulation reduces round-off across millions of grid cells.
        self.n = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_y = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_y2 = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_sq_error = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_abs_error = torch.zeros(self.channels, dtype=torch.float64)

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Add one batch after filtering invalid/land pixels."""
        mask = mask.expand_as(target).bool()
        for channel in range(self.channels):
            # Evaluation does not need gradients; detach and move compact
            # selected vectors to CPU before updating the totals.
            y = target[:, channel][mask[:, channel]].detach().double().cpu()
            p = prediction[:, channel][mask[:, channel]].detach().double().cpu()
            error = p - y
            self.n[channel] += y.numel()
            self.sum_y[channel] += y.sum()
            self.sum_y2[channel] += y.square().sum()
            self.sum_sq_error[channel] += error.square().sum()
            self.sum_abs_error[channel] += error.abs().sum()

    def compute(self) -> dict[str, list[float]]:
        """Convert accumulated sums into one value per target channel.

        R² compares squared model error with the variance of the true field:
        1 is perfect, 0 matches predicting the mean, and negative values are
        worse than that mean baseline.
        """
        n = self.n.clamp_min(1)
        total_variance = self.sum_y2 - self.sum_y.square() / n
        return {
            "rmse": torch.sqrt(self.sum_sq_error / n).tolist(),
            "mae": (self.sum_abs_error / n).tolist(),
            "r2": (1 - self.sum_sq_error / total_variance.clamp_min(1e-12)).tolist(),
        }
