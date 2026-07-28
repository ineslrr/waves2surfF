"""Dataset that pairs wave inputs and current targets via pluggable sources."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .daily_index import SamplePair, build_split_pairs
from .data import ChannelStats
from .sources import DataSource, get_source


class WaveCurrentDataset(Dataset):
    """Pair wave fields and surface currents using two :class:`DataSource`s.

    Layout-specific path discovery and field I/O live in
    ``ocean_velocity.sources`` (selected by ``sources.*.layout`` in JSON).
    """
    DERIVED_VARIABLES = {"KX", "KY"} # adding flexibility to derive new input fields ---> see read_wave_variable

    def __init__(
        self,
        pairs: Sequence[SamplePair],
        input_variables: Sequence[str],
        target_variables: Sequence[str],
        wave_source: DataSource,
        current_source: DataSource,
        *,
        bbox: dict[str, float] | None = None,
        metadata_variables: Sequence[str] = (),
        calendar_features: Sequence[str] = (),
        valid_mask_variable: str | None = None,
        input_stats: ChannelStats | None = None,
        metadata_stats: ChannelStats | None = None,
        target_stats: ChannelStats | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("WaveCurrentDataset received an empty sample list")
        if metadata_variables:
            raise ValueError(
                "WaveCurrentDataset does not read file metadata variables yet; "
                "use calendar_features only, or the single-file NetCDFFieldDataset"
            )
        self.pairs = list(pairs)
        self.input_variables = tuple(input_variables)
        self.target_variables = tuple(target_variables)
        self.wave_source = wave_source
        self.current_source = current_source
        self.bbox = bbox
        self.metadata_variables = tuple(metadata_variables)
        self.calendar_features = tuple(calendar_features)
        self.valid_mask_variable = valid_mask_variable
        self.input_stats = input_stats
        self.metadata_stats = metadata_stats
        self.target_stats = target_stats

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split: str,
        *,
        input_stats: ChannelStats | None = None,
        metadata_stats: ChannelStats | None = None,
        target_stats: ChannelStats | None = None,
    ) -> "WaveCurrentDataset":
        """Build a split dataset from a dual-source experiment configuration."""
        data = config["data"]
        sources = data["sources"]
        wave_cfg = sources["waves"]
        current_cfg = sources["currents"]
        wave_source = get_source(wave_cfg.get("layout", "daily_tree"), wave_cfg)
        current_source = get_source(
            current_cfg.get("layout", "daily_tree"), current_cfg
        )
        by_split = build_split_pairs(
            wave_source,
            current_source,
            data["splits"],
            match_tolerance_hours=float(data.get("match_tolerance_hours", 3.0)),
        )
        if split not in by_split:
            raise KeyError(
                f"Unknown split {split!r}; available: {sorted(by_split)}"
            )
        input_variables = data.get("input_variables", wave_cfg.get("variables"))
        target_variables = data.get(
            "target_variables", current_cfg.get("variables")
        )
        return cls(
            by_split[split],
            input_variables,
            target_variables,
            wave_source,
            current_source,
            bbox=data.get("bbox"),
            metadata_variables=data.get("metadata_variables", []),
            calendar_features=data.get("calendar_features", []),
            valid_mask_variable=data.get("valid_mask_variable"),
            input_stats=input_stats,
            metadata_stats=metadata_stats,
            target_stats=target_stats,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _calendar_values(self, when: datetime) -> list[float]:
        if not self.calendar_features:
            return []
        year_length = 365.2425
        day = float(when.timetuple().tm_yday) - 1.0
        hour = float(when.hour) + float(when.minute) / 60.0
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

    @staticmethod
    def _normalize(values: np.ndarray, stats: ChannelStats | None) -> np.ndarray:
        if stats is None:
            return values
        mean = np.asarray(stats.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(stats.std, dtype=np.float32)[:, None, None]
        if len(mean) != values.shape[0]:
            raise ValueError("Normalization statistics do not match channel count")
        return (values - mean) / np.maximum(std, 1e-8)

        
    def _read_wave_variable(self, pair, name, cache):
        # print("READ REQUEST:", repr(name))
    
        if name in cache:
            return cache[name]
    
        if name in self.DERIVED_VARIABLES:
            # print("COMPUTING DERIVED:", name)
    
            T = self._read_wave_variable(pair, "VTM10", cache)
            theta = np.deg2rad(
                self._read_wave_variable(pair, "VMDR", cache)
            )
    
            L = 9.81 * T**2 / (2 * np.pi)
            k = 2 * np.pi / L
    
            cache["KX"] = k * np.cos(theta)
            cache["KY"] = k * np.sin(theta)
    
            return cache[name]
    
        cache[name] = self.wave_source.read_field(
            pair.wave,
            name,
            self.bbox,
        )
        return cache[name]


    
    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[item]
        targets = [
            self.current_source.read_field(pair.current, name, self.bbox)
            for name in self.target_variables
        ]
        
        # inputs = [
        #     self.wave_source.read_field(pair.wave, name, self.bbox)
        #     for name in self.input_variables
        # ]
        
        cache = {}
        inputs = [
            self._read_wave_variable(pair, name, cache)
            for name in self.input_variables
        ]

        
        if inputs[0].shape != targets[0].shape:
            raise ValueError(
                "Wave and current spatial shapes differ after bbox crop: "
                f"{inputs[0].shape} vs {targets[0].shape}. "
                "v1 does not regrid; use matching grids or adjust bbox."
            )
        x_raw = np.stack(inputs)
        y_raw = np.stack(targets)
        metadata_raw = np.asarray(
            self._calendar_values(pair.wave.time), dtype=np.float32
        )

        valid = np.isfinite(x_raw).all(axis=0) & np.isfinite(y_raw).all(axis=0)
        metadata_valid = bool(
            metadata_raw.size == 0 or np.isfinite(metadata_raw).all()
        )
        if self.valid_mask_variable:
            supplied = self.wave_source.read_field(
                pair.wave, self.valid_mask_variable, self.bbox
            )
            valid &= np.isfinite(supplied) & (supplied > 0)

        x = self._normalize(np.nan_to_num(x_raw), self.input_stats)
        y = self._normalize(np.nan_to_num(y_raw), self.target_stats)
        if metadata_raw.size:
            metadata = self._normalize(
                np.nan_to_num(metadata_raw)[:, None, None], self.metadata_stats
            )[:, 0, 0]
        else:
            metadata = np.zeros((0,), dtype=np.float32)
        if not metadata_valid:
            valid[:] = False
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "metadata": torch.from_numpy(metadata.astype(np.float32)),
            "metadata_valid": torch.tensor(metadata_valid),
            "valid_mask": torch.from_numpy(valid[None]),
            "index": torch.tensor(item),
        }
