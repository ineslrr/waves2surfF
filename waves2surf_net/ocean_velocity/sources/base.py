"""Abstract interface and registry for wave / current file layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Type

import numpy as np

from ..daily_index import TimeStampRef

_REGISTRY: dict[str, Type["DataSource"]] = {}


def register_source(name: str, cls: Type["DataSource"]) -> Type["DataSource"]:
    """Register ``cls`` under ``name`` (also usable as a class decorator)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("layout name must be non-empty")
    _REGISTRY[key] = cls
    return cls


def get_source(layout: str, config: dict[str, Any]) -> "DataSource":
    """Instantiate the source registered for ``layout`` with ``config``."""
    key = layout.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown data layout {layout!r}. Known layouts: {known}. "
            "Register a DataSource subclass or fix config sources.*.layout."
        )
    return _REGISTRY[key](config)


def list_layouts() -> list[str]:
    """Return registered layout names."""
    return sorted(_REGISTRY)


class DataSource(ABC):
    """One on-disk product layout (waves, currents, …).

    Subclasses own path discovery and field I/O. Pairing / splits / normalization
    stay in :class:`~ocean_velocity.wavecurrentdataset.WaveCurrentDataset`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if "dir" not in config:
            raise ValueError("source config requires 'dir'")
        self.config = dict(config)
        self.root = config["dir"]
        self.time_variable = config.get("time_variable", "time")

    @abstractmethod
    def list_timestamps(
        self, start: datetime, end: datetime
    ) -> list[TimeStampRef]:
        """Return every timestep in ``[start, end]`` (inclusive by calendar day)."""

    @abstractmethod
    def read_field(
        self,
        ref: TimeStampRef,
        variable: str,
        bbox: dict[str, float] | None,
    ) -> np.ndarray:
        """Return a 2-D ``float32`` field for ``variable`` at ``ref``."""

    def __getstate__(self) -> dict[str, Any]:
        """Default pickle state; subclasses should drop open file handles."""
        return {"config": self.config, "root": self.root, "time_variable": self.time_variable}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.config = state["config"]
        self.root = state["root"]
        self.time_variable = state["time_variable"]
