"""Daily NetCDF tree layout: ``{dir}/{YYYY}/{MM}/{YYYYMMDD}.nc``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..daily_index import (
    TimeStampRef,
    _to_python_datetime,
    daily_nc_path,
    iter_days,
)
from .base import DataSource, ensure_south_to_north


class DailyTreeSource(DataSource):
    """MFWAM / Copernicus-style one NetCDF file per calendar day."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._handles: dict[str, Any] = {}
        self._handle_order: list[str] = []
        self._bbox_slices: dict[str, tuple[slice, slice]] = {}
        self.max_open_files = int(config.get("max_open_files", 4))

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["max_open_files"] = self.max_open_files
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.max_open_files = state.get("max_open_files", 4)
        self._handles = {}
        self._handle_order = []
        self._bbox_slices = {}

    def list_timestamps(
        self, start: datetime, end: datetime
    ) -> list[TimeStampRef]:
        from netCDF4 import Dataset, num2date

        root = Path(self.root)
        refs: list[TimeStampRef] = []
        for day in iter_days(start, end):
            path = daily_nc_path(root, day)
            if not path.is_file():
                continue
            with Dataset(path, "r") as ds:
                if self.time_variable not in ds.variables:
                    raise KeyError(
                        f"{path}: missing time variable {self.time_variable!r}"
                    )
                time_var = ds.variables[self.time_variable]
                values = num2date(
                    time_var[:],
                    units=time_var.units,
                    calendar=getattr(time_var, "calendar", "standard"),
                )
                if not isinstance(values, (list, tuple, np.ndarray)):
                    values = [values]
                for index, raw in enumerate(values):
                    refs.append(
                        TimeStampRef(
                            time=_to_python_datetime(raw),
                            path=str(path),
                            time_index=int(index),
                        )
                    )
        refs.sort(key=lambda item: item.time)
        return refs

    def _open(self, path: str):
        if path in self._handles:
            if path in self._handle_order:
                self._handle_order.remove(path)
            self._handle_order.append(path)
            return self._handles[path]
        from netCDF4 import Dataset

        handle = Dataset(path, "r")
        self._handles[path] = handle
        self._handle_order.append(path)
        while len(self._handle_order) > self.max_open_files:
            old = self._handle_order.pop(0)
            old_handle = self._handles.pop(old, None)
            if old_handle is not None:
                old_handle.close()
            self._bbox_slices.pop(old, None)
        return handle

    @staticmethod
    def _lat_lon_names(ds) -> tuple[str, str]:
        if "latitude" in ds.variables and "longitude" in ds.variables:
            return "latitude", "longitude"
        if "lat" in ds.variables and "lon" in ds.variables:
            return "lat", "lon"
        raise KeyError(f"No latitude/longitude coordinates in {ds.filepath()}")

    def _bbox_slices_for(
        self, path: str, bbox: dict[str, float] | None
    ) -> tuple[slice, slice]:
        if bbox is None:
            return slice(None), slice(None)
        cache_key = path
        if cache_key in self._bbox_slices:
            return self._bbox_slices[cache_key]
        ds = self._open(path)
        lat_name, lon_name = self._lat_lon_names(ds)
        lats = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lons = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        lat_min = float(bbox["lat_min"])
        lat_max = float(bbox["lat_max"])
        lon_min = float(bbox["lon_min"])
        lon_max = float(bbox["lon_max"])
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if lon_min <= lon_max:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
        else:
            lon_mask = (lons >= lon_min) | (lons <= lon_max)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError(f"bbox {bbox} selects an empty region in {path}")
        ys = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
        xs = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
        if not np.all(lat_mask[ys]):
            raise ValueError(
                f"Non-contiguous latitude mask for bbox in {path}"
            )
        if not np.all(lon_mask[xs]) and lon_min <= lon_max:
            raise ValueError(
                f"Non-contiguous longitude mask for bbox in {path}"
            )
        self._bbox_slices[cache_key] = (ys, xs)
        return ys, xs

    def read_field(
        self,
        ref: TimeStampRef,
        variable: str,
        bbox: dict[str, float] | None,
    ) -> np.ndarray:
        ds = self._open(ref.path)
        if variable not in ds.variables:
            raise KeyError(f"{ref.path}: missing variable {variable!r}")
        var = ds.variables[variable]
        ys, xs = self._bbox_slices_for(ref.path, bbox)
        if var.ndim == 4:
            value = var[ref.time_index, 0, ys, xs]
        elif var.ndim == 3:
            value = var[ref.time_index, ys, xs]
        elif var.ndim == 2:
            value = var[ys, xs]
        else:
            raise ValueError(
                f"Variable {variable!r} in {ref.path} has unsupported "
                f"ndim={var.ndim}"
            )
        value = np.asarray(np.ma.filled(value, np.nan), dtype=np.float32)
        lat_name, _ = self._lat_lon_names(ds)
        lats = np.asarray(ds.variables[lat_name][ys], dtype=np.float64)
        return ensure_south_to_north(value, lats)
