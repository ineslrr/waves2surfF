"""Public package interface for the ocean surface-velocity project.

Most users construct models through the configuration-driven training code,
but exporting :class:`Waves2SurfNet` here also makes interactive use concise:
``from ocean_velocity import Waves2SurfNet``.
"""

from .model import UNet, Waves2SurfNet
from .diffusion import EDMPreconditioner, edm_loss, edm_sample
from .diffusion_model import ConditionalDiffusionUNet

# ``__all__`` documents the intentionally supported top-level API; internal
# helpers remain available from their individual modules.
__all__ = [
    "Waves2SurfNet",
    "UNet",
    "ConditionalDiffusionUNet",
    "EDMPreconditioner",
    "edm_loss",
    "edm_sample",
]
