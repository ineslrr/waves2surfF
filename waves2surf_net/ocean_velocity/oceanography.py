"""Oceanographic diagnostics computed from vector fields."""

from __future__ import annotations

import numpy as np


def compute_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Compute relative vorticity.

    ζ = ∂v/∂x − ∂u/∂y

    Parameters
    ----------
    u
        Eastward velocity component.
    v
        Northward velocity component.
    dx
        Grid spacing in the x direction.
    dy
        Grid spacing in the y direction.

    Returns
    -------
    np.ndarray
        Relative vorticity.
    """
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy


def compute_divergence(
    u: np.ndarray,
    v: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Compute horizontal divergence.

    ∇·V = ∂u/∂x + ∂v/∂y

    Parameters
    ----------
    u
        Eastward velocity component.
    v
        Northward velocity component.
    dx
        Grid spacing in the x direction.
    dy
        Grid spacing in the y direction.

    Returns
    -------
    np.ndarray
        Horizontal divergence.
    """
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return du_dx + dv_dy