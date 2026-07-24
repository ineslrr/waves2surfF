"""Training objectives for complementary notions of a good velocity field.

The pointwise term asks for accurate velocity at each wet pixel. Optional
spectral and gradient terms emphasize multiscale energy and derived flow
structure. ``ocean_velocity.train._loss`` combines these terms using weights
from the experiment configuration.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over valid pixels and output channels only.

    Dividing by the mask count, rather than averaging a zero-filled image,
    prevents land fraction from changing the effective scale of the loss.
    """
    # One [B,1,H,W] ocean mask applies to both u and v components.
    mask = valid_mask.expand_as(prediction)
    error = torch.abs(prediction - target)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare predicted and target 2-D kinetic-energy spectra.

    This auxiliary objective discourages an overly smooth prediction that has
    acceptable pixel error but too little small-scale energy. It assumes the
    two output channels are the horizontal velocity components ``u`` and ``v``.
    """
    if prediction.shape[1] != 2:
        raise ValueError("Spectral kinetic-energy loss requires two output channels")
    height, width = prediction.shape[-2:]
    # Tapering reduces artificial high-frequency energy caused by a mismatch
    # between opposite boundaries in the discrete Fourier transform.
    wy = torch.hann_window(height, device=prediction.device)
    wx = torch.hann_window(width, device=prediction.device)
    window = torch.outer(wy, wx)[None, None]
    mask = valid_mask.to(prediction.dtype)

    def energy(value: torch.Tensor) -> torch.Tensor:
        """Return kinetic energy at every resolved 2-D Fourier mode."""
        transform = torch.fft.rfft2(value * mask * window, dim=(-2, -1))
        return 0.5 * transform.abs().square().sum(dim=1)

    # Log energy prevents only the largest spatial scales from dominating.
    return F.mse_loss(
        torch.log(energy(prediction) + eps),
        torch.log(energy(target) + eps),
    )


def velocity_gradients(
    velocity: torch.Tensor, dx: float = 1.0, dy: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate ``du/dx, du/dy, dv/dx, dv/dy`` with centered differences.

    The one-cell border is omitted because a centered stencil requires a
    neighbor on both sides. ``dx`` and ``dy`` must describe the physical grid
    spacing if these derivatives are meant to have physical units.
    """
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
    """Penalize errors in vorticity and horizontal divergence.

    Pixel losses constrain velocity itself; this term additionally asks the
    prediction to reproduce rotation (``dv/dx - du/dy``) and convergence
    (``du/dx + dv/dy``), which are often scientifically important diagnostics.
    """
    pred = velocity_gradients(prediction, dx, dy)
    true = velocity_gradients(target, dx, dy)
    mask = valid_mask[..., 1:-1, 1:-1]
    # Construct (vorticity, divergence) from the four component derivatives.
    pred_fields = (pred[2] - pred[1], pred[0] + pred[3])
    true_fields = (true[2] - true[1], true[0] + true[3])
    return sum(masked_l1(a, b, mask) for a, b in zip(pred_fields, true_fields))
