"""Yearly multi-time NetCDF layout: ``{dir}/cur_{YYYY}.nc``.

Typical Mercator surface currents after ``ncrcat`` of hourly files::

    uo(time, depth, latitude, longitude)
    vo(time, depth, latitude, longitude)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..daily_index import TimeStampRef, _to_python_datetime
from .base import DataSource


class YearlyNetCDFSource(DataSource):
    """One NetCDF file per calendar year with a long ``time`` axis."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Support "{year}" or legacy "{YYYY}" in the pattern.
        pattern = config.get("file_pattern", "cur_{year}.nc")
        self.file_pattern = pattern.replace("{YYYY}", "{year}")
        self.depth_index = int(config.get("depth_index", 0))
        self._handles: dict[str, Any] = {}
        self._handle_order: list[str] = []
        self._bbox_slices: dict[str, tuple[slice, slice]] = {}
        self._times_cache: dict[str, np.ndarray] = {}
        self.max_open_files = int(config.get("max_open_files", 2))

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.update(
            {
                "file_pattern": self.file_pattern,
                "depth_index": self.depth_index,
                "max_open_files": self.max_open_files,
            }
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.file_pattern = state["file_pattern"]
        self.depth_index = state["depth_index"]
        self.max_open_files = state.get("max_open_files", 2)
        self._handles = {}
        self._handle_order = []
        self._bbox_slices = {}
        self._times_cache = {}

    def _year_path(self, year: int) -> Path:
        return Path(self.root) / self.file_pattern.format(year=year)

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
            self._times_cache.pop(old, None)
        return handle

    def _read_time_array(self, path: str) -> np.ndarray:
        """Return timezone-naive datetime64[s] array for the file time axis."""
        if path in self._times_cache:
            return self._times_cache[path]
        from netCDF4 import num2date

        ds = self._open(path)
        if self.time_variable not in ds.variables:
            raise KeyError(f"{path}: missing time variable {self.time_variable!r}")
        time_var = ds.variables[self.time_variable]
        values = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]
        py_times = [_to_python_datetime(v) for v in values]
        arr = np.asarray(py_times, dtype="datetime64[s]")
        self._times_cache[path] = arr
        return arr

    def list_timestamps(
        self, start: datetime, end: datetime
    ) -> list[TimeStampRef]:
        refs: list[TimeStampRef] = []
        start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end.replace(hour=23, minute=59, second=59, microsecond=0)
        start64 = np.datetime64(start_day, "s")
        end64 = np.datetime64(end_day, "s")

        for year in range(start.year, end.year + 1):
            path = self._year_path(year)
            if not path.is_file():
                continue
            path_str = str(path)
            times = self._read_time_array(path_str)
            # Inclusive calendar-day window on the wave/current pairing side.
            mask = (times >= start64) & (times <= end64)
            indices = np.where(mask)[0]
            for index in indices:
                t = times[int(index)]
                if isinstance(t, np.datetime64):
                    t = t.astype("datetime64[s]").item()
                refs.append(
                    TimeStampRef(
                        time=t,
                        path=path_str,
                        time_index=int(index),
                    )
                )
        refs.sort(key=lambda item: item.time)
        return refs

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
        if path in self._bbox_slices:
            return self._bbox_slices[path]
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
            raise ValueError(f"Non-contiguous latitude mask for bbox in {path}")
        if not np.all(lon_mask[xs]) and lon_min <= lon_max:
            raise ValueError(f"Non-contiguous longitude mask for bbox in {path}")
        self._bbox_slices[path] = (ys, xs)
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
        # Expected: (time, depth, lat, lon) or (time, lat, lon)
        if var.ndim == 4:
            value = var[ref.time_index, self.depth_index, ys, xs]
        elif var.ndim == 3:
            value = var[ref.time_index, ys, xs]
        elif var.ndim == 2:
            value = var[ys, xs]
        else:
            raise ValueError(
                f"Variable {variable!r} in {ref.path} has unsupported "
                f"ndim={var.ndim}"
            )
        return np.asarray(np.ma.filled(value, np.nan), dtype=np.float32)
