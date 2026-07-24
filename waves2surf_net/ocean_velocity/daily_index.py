"""Index and pair daily wave / current NetCDF trees by timestamp.

Expected on-disk layout (same as download_mfwam / download_surface_currents)::

    {root}/{YYYY}/{MM}/{YYYYMMDD}.nc
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TimeStampRef:
    """One timestep inside a daily NetCDF file."""

    time: datetime
    path: str
    time_index: int


@dataclass(frozen=True)
class SamplePair:
    """One training sample: a wave time paired with a current time."""

    wave: TimeStampRef
    current: TimeStampRef


def parse_yyyymmdd(value: str) -> datetime:
    """Parse ``YYYYMMDD`` or ``YYYY-MM-DD`` into a datetime at midnight."""
    text = value.replace("-", "")
    return datetime.strptime(text, "%Y%m%d")


def daily_nc_path(root: Path, day: datetime) -> Path:
    """Return ``{root}/{YYYY}/{MM}/{YYYYMMDD}.nc``."""
    return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.strftime('%Y%m%d')}.nc"


def iter_days(start: datetime, end: datetime) -> Iterable[datetime]:
    """Yield calendar days from ``start`` through ``end`` inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _to_python_datetime(value: Any) -> datetime:
    """Convert netCDF4 / cftime / numpy times to naive UTC ``datetime``."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "timetuple"):
        # cftime objects expose year/month/day/hour/...
        return datetime(
            int(value.year),
            int(value.month),
            int(value.day),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
        )
    raise TypeError(f"Unsupported time type: {type(value)!r}")


def pair_timestamps(
    waves: Sequence[TimeStampRef],
    currents: Sequence[TimeStampRef],
    *,
    tolerance_hours: float = 3.0,
) -> list[SamplePair]:
    """Match each wave time to the nearest current time within ``tolerance_hours``."""
    if not waves or not currents:
        return []
    current_times = np.asarray(
        [ref.time.timestamp() for ref in currents], dtype=np.float64
    )
    tolerance_s = float(tolerance_hours) * 3600.0
    pairs: list[SamplePair] = []
    for wave in waves:
        wave_t = wave.time.timestamp()
        index = int(np.searchsorted(current_times, wave_t))
        candidates = []
        if 0 <= index < len(currents):
            candidates.append(index)
        if index - 1 >= 0:
            candidates.append(index - 1)
        best = None
        best_dt = None
        for candidate in candidates:
            dt = abs(current_times[candidate] - wave_t)
            if best_dt is None or dt < best_dt:
                best = candidate
                best_dt = dt
        if best is None or best_dt is None or best_dt > tolerance_s:
            continue
        pairs.append(SamplePair(wave=wave, current=currents[best]))
    return pairs


def resolve_split_ranges(
    splits: dict[str, Any],
) -> dict[str, tuple[datetime, datetime]]:
    """Normalize split config into ``{name: (start, end)}`` datetimes.

    Supported forms:

    - Explicit ranges: ``{"train": {"start": "20230101", "end": "20231231"}, ...}``
    - Shorthand: ``{"train": {...}, "evaluation": {...}}`` → evaluation is
      split 20% / 80% by calendar span into ``validation`` / ``test``.
    """
    if "evaluation" in splits and (
        "validation" not in splits or "test" not in splits
    ):
        if "train" not in splits:
            raise ValueError("splits.evaluation requires splits.train")
        train = splits["train"]
        evaluation = splits["evaluation"]
        eval_start = parse_yyyymmdd(evaluation["start"])
        eval_end = parse_yyyymmdd(evaluation["end"])
        if eval_end < eval_start:
            raise ValueError("evaluation end must be >= start")
        span_days = (eval_end - eval_start).days + 1
        val_days = max(1, int(round(span_days * 0.2)))
        val_end = eval_start + timedelta(days=val_days - 1)
        if val_end >= eval_end:
            val_end = eval_end - timedelta(days=1)
        test_start = val_end + timedelta(days=1)
        return {
            "train": (parse_yyyymmdd(train["start"]), parse_yyyymmdd(train["end"])),
            "validation": (eval_start, val_end),
            "test": (test_start, eval_end),
        }

    resolved: dict[str, tuple[datetime, datetime]] = {}
    for name, value in splits.items():
        if name == "evaluation":
            continue
        if not isinstance(value, dict) or "start" not in value or "end" not in value:
            raise ValueError(
                f"splits.{name} must be a date range object with start/end "
                f"(YYYYMMDD), got {value!r}"
            )
        start = parse_yyyymmdd(value["start"])
        end = parse_yyyymmdd(value["end"])
        if end < start:
            raise ValueError(f"splits.{name}: end must be >= start")
        resolved[name] = (start, end)
    return resolved


def build_split_pairs(
    wave_source: Any,
    current_source: Any,
    splits: dict[str, Any],
    *,
    match_tolerance_hours: float = 3.0,
) -> dict[str, list[SamplePair]]:
    """Build paired sample lists using two :class:`DataSource` backends."""
    ranges = resolve_split_ranges(splits)
    if not ranges:
        raise ValueError("No date-range splits found in configuration")

    overall_start = min(start for start, _ in ranges.values())
    overall_end = max(end for _, end in ranges.values())

    waves = wave_source.list_timestamps(overall_start, overall_end)
    currents = current_source.list_timestamps(overall_start, overall_end)
    all_pairs = pair_timestamps(
        waves, currents, tolerance_hours=match_tolerance_hours
    )

    by_split: dict[str, list[SamplePair]] = {name: [] for name in ranges}
    for pair in all_pairs:
        day = pair.wave.time.replace(hour=0, minute=0, second=0, microsecond=0)
        for name, (start, end) in ranges.items():
            if start <= day <= end:
                by_split[name].append(pair)
                break
    return by_split


def list_timestamps(
    root: str | Path,
    start: datetime,
    end: datetime,
    *,
    time_variable: str = "time",
) -> list[TimeStampRef]:
    """Backward-compatible helper for the daily-tree layout."""
    from .sources import get_source

    source = get_source(
        "daily_tree", {"dir": str(root), "time_variable": time_variable}
    )
    return source.list_timestamps(start, end)