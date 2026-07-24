"""Public package interface for the ocean surface-velocity project.

Most users construct models through the configuration-driven training code,
but exporting :class:`UNet` here also makes interactive use concise:
``from ocean_velocity import UNet``.
"""

from .model import UNet

# ``__all__`` documents the intentionally supported top-level API; internal
# helpers remain available from their individual modules.
__all__ = ["UNet"]
