"""Pluggable on-disk data sources for WaveCurrentDataset.

Each layout implements :class:`DataSource` and registers a name used in JSON::

    "waves": {"dir": "...", "layout": "daily_tree", ...}
    "currents": {"dir": "...", "layout": "my_currents", ...}

Add a new loader by:

1. Creating a class that subclasses :class:`DataSource`
2. Implementing ``list_timestamps`` and ``read_field``
3. Calling ``register_source("my_currents", MyCurrentsSource)``
4. Setting ``"layout": "my_currents"`` in the config

See :mod:`ocean_velocity.sources.daily_tree` for the reference implementation
and :mod:`ocean_velocity.sources.custom_template` for a stub to copy.
"""

from __future__ import annotations

from .base import DataSource, get_source, list_layouts, register_source
from .daily_tree import DailyTreeSource
from .yearly_nc import YearlyNetCDFSource

# Built-in layouts (name used in config["layout"]).
register_source("daily_tree", DailyTreeSource)
register_source("mfwam", DailyTreeSource)  # alias for wave trees
register_source("yearly_nc", YearlyNetCDFSource)
register_source("mercator_yearly", YearlyNetCDFSource)

__all__ = [
    "DataSource",
    "DailyTreeSource",
    "YearlyNetCDFSource",
    "get_source",
    "list_layouts",
    "register_source",
]
