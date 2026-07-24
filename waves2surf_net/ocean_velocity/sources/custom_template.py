"""Template for a custom currents (or waves) layout.

Copy this file to e.g. ``my_currents.py``, implement the two methods, then
register it at the bottom of ``sources/__init__.py``::

    from .my_currents import MyCurrentsSource
    register_source("my_currents", MyCurrentsSource)

Config::

    "currents": {
      "dir": "/path/to/data",
      "layout": "my_currents",
      "variables": ["uo", "vo"]
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ..daily_index import TimeStampRef
from .base import DataSource


class CustomSourceTemplate(DataSource):
    """Replace this class with your real layout-specific loader."""

    def list_timestamps(
        self, start: datetime, end: datetime
    ) -> list[TimeStampRef]:
        """Discover files under ``self.root`` and return timestep references.

        Each :class:`TimeStampRef` must provide:

        - ``time``: Python ``datetime`` of the snapshot
        - ``path``: file path your ``read_field`` will open
        - ``time_index``: index inside that file (0 if one time per file)
        """
        raise NotImplementedError(
            f"{type(self).__name__}.list_timestamps is not implemented. "
            "Copy sources/custom_template.py and register your layout."
        )

    def read_field(
        self,
        ref: TimeStampRef,
        variable: str,
        bbox: dict[str, float] | None,
    ) -> np.ndarray:
        """Load one 2-D float32 field, cropped to ``bbox`` if provided.

        ``bbox`` keys: lon_min, lon_max, lat_min, lat_max.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.read_field is not implemented. "
            "Copy sources/custom_template.py and register your layout."
        )
