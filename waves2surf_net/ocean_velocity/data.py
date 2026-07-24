"""Turn NetCDF snapshots into tensors consumed by the training pipeline.

The dataset is the boundary between scientific data conventions and PyTorch:
it selects variables/times, builds masks, standardizes channels, and returns
consistently shaped tensors.  Keeping that work here prevents the model and
training loop from becoming tied to one particular NetCDF file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel mean and standard deviation used for standardization.

    Standardizing variables with different physical scales (for example SSH
    in metres and wind in m/s) usually makes gradient-based optimization much
    easier. Statistics must be estimated from the training split only.
    """
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelStats":
        """Construct typed statistics from the corresponding JSON object."""
        return cls(tuple(value["mean"]), tuple(value["std"]))

    def to_dict(self) -> dict[str, list[float]]:
        """Convert statistics back to a JSON-serializable representation."""
        return {"mean": list(self.mean), "std": list(self.std)}


class NetCDFFieldDataset(Dataset):
    """Read aligned 2-D inputs and vector targets from one NetCDF file.

    Spatial input variables may have ``[time, y, x]`` or ``[y, x]``
    dimensions. Metadata variables may be scalars or ``[time]`` series.
    Calendar metadata is derived from the NetCDF time coordinate.
    A file handle is opened lazily in each DataLoader worker.
    """

    def __init__(
        self,
        path: str | Path,
        input_variables: Sequence[str],
        target_variables: Sequence[str],
        indices: Sequence[int],
        *,
        metadata_variables: Sequence[str] = (),
        time_variable: str = "time",
        calendar_features: Sequence[str] = (),
        valid_mask_variable: str | None = None,
        spatial_slice: Sequence[int] | None = None,
        input_stats: ChannelStats | None = None,
        metadata_stats: ChannelStats | None = None,
        target_stats: ChannelStats | None = None,
    ) -> None:
        """Store the data recipe without opening the potentially large file.

        ``indices`` defines the time split, while ``spatial_slice`` optionally
        restricts all samples to one rectangular region. File access is
        deliberately deferred until a sample is requested.
        """
        self.path = str(path)
        self.input_variables = tuple(input_variables)
        self.target_variables = tuple(target_variables)
        self.metadata_variables = tuple(metadata_variables)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.time_variable = time_variable
        self.calendar_features = tuple(calendar_features)
        self.valid_mask_variable = valid_mask_variable
        self.spatial_slice = tuple(spatial_slice) if spatial_slice else None
        self.input_stats = input_stats
        self.metadata_stats = metadata_stats
        self.target_stats = target_stats
        self._dataset = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open NetCDF handle when PyTorch serializes a worker copy.

        DataLoader processes must not share a NetCDF handle inherited from the
        parent process. Each worker will reopen the file lazily through ``nc``.
        """
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    @property
    def nc(self):
        """Return this worker's lazily opened read-only NetCDF handle."""
        if self._dataset is None:
            from netCDF4 import Dataset

            self._dataset = Dataset(self.path, "r")
        return self._dataset

    def __len__(self) -> int:
        """Report the number of configured snapshots in this split."""
        return len(self.indices)

    def _slices(self) -> tuple[slice, slice]:
        """Translate the optional ``[y0, y1, x0, x1]`` box into NumPy slices."""
        if self.spatial_slice is None:
            return slice(None), slice(None)
        y0, y1, x0, x1 = self.spatial_slice
        return slice(y0, y1), slice(x0, x1)

    def _read_field(self, name: str, time_index: int, shape=None) -> np.ndarray:
        """Read one spatial channel and return a 2-D ``float32`` array.

        Time-varying fields are indexed at ``time_index``; static fields are
        reused for every sample. One-dimensional coordinates and scalars can
        be expanded over a known target grid.
        """
        variable = self.nc.variables[name]
        ys, xs = self._slices()
        # The baseline assumes conventional [time, y, x] ordering.
        if variable.ndim >= 3:
            value = variable[time_index, ys, xs]
        elif variable.ndim == 2:
            value = variable[ys, xs]
        elif variable.ndim == 1:
            if self.time_variable in variable.dimensions:
                value = variable[time_index]
            else:
                value = np.asarray(variable[:], dtype=np.float32)
                if shape is None:
                    raise ValueError(
                        f"Cannot broadcast 1-D spatial variable {name!r} "
                        "before grid shape is known"
                    )
                if value.size == shape[0]:
                    value = np.broadcast_to(value[:, None], shape)
                elif value.size == shape[1]:
                    value = np.broadcast_to(value[None, :], shape)
                else:
                    raise ValueError(
                        f"Spatial coordinate {name!r} does not match {shape}"
                    )
        else:
            value = variable[...]
        # Converting NetCDF masked values to NaN lets one common finite-value
        # test create the training mask later.
        value = np.asarray(np.ma.filled(value, np.nan), dtype=np.float32)
        if value.ndim == 0:
            if shape is None:
                raise ValueError(f"Cannot broadcast scalar {name!r} before grid shape is known")
            value = np.full(shape, value.item(), dtype=np.float32)
        return value

    def _calendar_values(self, time_index: int) -> list[float]:
        """Encode periodic time without an artificial year/day discontinuity.

        A raw day-of-year makes December 31 and January 1 look far apart.
        Sine/cosine pairs place them next to each other on a unit circle.
        """
        if not self.calendar_features:
            return []
        from netCDF4 import num2date

        time = self.nc.variables[self.time_variable]
        date = num2date(
            time[time_index],
            units=time.units,
            calendar=getattr(time, "calendar", "standard"),
        )
        year_length = 360 if getattr(time, "calendar", "") == "360_day" else 365.2425
        day = float(date.timetuple().tm_yday) - 1.0
        hour = float(date.hour) + float(date.minute) / 60.0
        values = {
            "year_sin": np.sin(2 * np.pi * day / year_length),
            "year_cos": np.cos(2 * np.pi * day / year_length),
            "day_sin": np.sin(2 * np.pi * hour / 24.0),
            "day_cos": np.cos(2 * np.pi * hour / 24.0),
        }
        unknown = set(self.calendar_features) - values.keys()
        if unknown:
            raise ValueError(f"Unknown calendar features: {sorted(unknown)}")
        return [float(values[name]) for name in self.calendar_features]

    def _read_metadata(self, name: str, time_index: int) -> float:
        """Read one global scalar associated with the current snapshot."""
        variable = self.nc.variables[name]
        if variable.ndim == 1 and self.time_variable in variable.dimensions:
            value = variable[time_index]
        elif variable.ndim == 0:
            value = variable[...]
        else:
            raise ValueError(
                f"Metadata variable {name!r} must be scalar or indexed only by "
                f"{self.time_variable!r}; got dimensions {variable.dimensions}"
            )
        return float(np.ma.filled(value, np.nan))

    @staticmethod
    def _normalize(values: np.ndarray, stats: ChannelStats | None) -> np.ndarray:
        """Apply channel-wise z-score normalization while preserving shape."""
        if stats is None:
            return values
        mean = np.asarray(stats.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(stats.std, dtype=np.float32)[:, None, None]
        if len(mean) != values.shape[0]:
            raise ValueError("Normalization statistics do not match channel count")
        # The floor prevents division by zero for nearly constant variables.
        return (values - mean) / np.maximum(std, 1e-8)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """Build one model-ready sample.

        Returns spatial input ``x[C,H,W]``, target velocity ``y[2,H,W]``,
        global ``metadata[D]``, and ``valid_mask[1,H,W]``. DataLoader adds the
        leading batch dimension used throughout the rest of the pipeline.
        """
        time_index = int(self.indices[item])

        # Read targets first: their grid establishes the shape used when a
        # scalar or 1-D coordinate must be expanded spatially.
        targets = [self._read_field(name, time_index) for name in self.target_variables]
        shape = targets[0].shape
        inputs = [
            self._read_field(name, time_index, shape) for name in self.input_variables
        ]
        x_raw = np.stack(inputs)
        y_raw = np.stack(targets)
        metadata_raw = np.asarray(
            [
                self._read_metadata(name, time_index)
                for name in self.metadata_variables
            ]
            + self._calendar_values(time_index),
            dtype=np.float32,
        )

        # A pixel is trainable only when every input and target channel exists.
        valid = np.isfinite(x_raw).all(axis=0) & np.isfinite(y_raw).all(axis=0)
        metadata_valid = np.isfinite(metadata_raw).all()
        if self.valid_mask_variable:
            supplied_mask = self._read_field(
                self.valid_mask_variable, time_index, shape
            )
            valid &= np.isfinite(supplied_mask) & (supplied_mask > 0)

        # NaNs can safely become zero after masking; replacing them keeps
        # invalid arithmetic from contaminating convolution outputs.
        x = self._normalize(np.nan_to_num(x_raw), self.input_stats)
        y = self._normalize(np.nan_to_num(y_raw), self.target_stats)
        metadata = self._normalize(
            np.nan_to_num(metadata_raw)[:, None, None], self.metadata_stats
        )[:, 0, 0]
        # Global metadata affects the entire sample, so if any element is
        # missing there are no trustworthy pixels in that snapshot.
        if not metadata_valid:
            valid[:] = False
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "metadata": torch.from_numpy(metadata.astype(np.float32)),
            "metadata_valid": torch.tensor(metadata_valid),
            "valid_mask": torch.from_numpy(valid[None]),
            "index": torch.tensor(time_index),
        }


def load_stats(path: str | Path) -> tuple[ChannelStats, ChannelStats]:
    """Load input/target statistics from a standalone statistics JSON file.

    This compatibility helper is useful for small external scripts. The main
    pipeline normally reads all normalization groups from the run config.
    """
    with open(path) as handle:
        payload = json.load(handle)
    return ChannelStats.from_dict(payload["input"]), ChannelStats.from_dict(
        payload["target"]
    )
