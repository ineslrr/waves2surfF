"""Convert meteorological wave direction into cartesian unit components.

MFWAM ``VMDR`` is the direction waves come *from*, in degrees, clockwise from
north (meteorological convention). Surface currents ``(uo, vo)`` use a
trigonometric frame: eastward ``uo`` / ``+x``, northward ``vo`` / ``+y``, with
angle counterclockwise from east.

The unit wave-propagation vector ``(kx, ky)`` points in the direction waves
travel *to*, in that same trigonometric frame, so it is directly comparable to
``(uo, vo)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[np.floating]


def vmdr_to_kx_ky(vmdr_deg: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Map VMDR (degrees, from-direction, CW from north) to ``(kx, ky)``.

    Steps
    -----
    1. Convert "from" to geographic "to" by adding 180°.
    2. Convert geographic bearing (CW from north) to trigonometric angle
       (CCW from east): ``θ_trig = π/2 − θ_geo``.
    3. ``kx = cos(θ_trig)``, ``ky = sin(θ_trig)`` (unit vector).

    Examples (propagation *to*):
    - VMDR = 0° (from N) → to S → ``(kx, ky) = (0, -1)``
    - VMDR = 90° (from E) → to W → ``(kx, ky) = (-1, 0)``
    - VMDR = 180° (from S) → to N → ``(kx, ky) = (0, +1)``
    - VMDR = 270° (from W) → to E → ``(kx, ky) = (+1, 0)``
    """
    vmdr = np.asarray(vmdr_deg, dtype=np.float64)
    # Geographic "to" direction: radians, clockwise from north.
    wave_to_geo = np.deg2rad(vmdr) + np.pi
    # Trigonometric polar angle of the (east, north) vector.
    wave_trig = np.pi / 2.0 - wave_to_geo
    kx = np.cos(wave_trig).astype(np.float32)
    ky = np.sin(wave_trig).astype(np.float32)
    return kx, ky


# Compact closed form of the same transform (kept for readability above):
#   kx = -sin(VMDR_rad),  ky = -cos(VMDR_rad)
