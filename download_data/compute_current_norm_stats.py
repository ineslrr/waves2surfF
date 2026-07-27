#!/usr/bin/env python3
"""
Compute regional mean / std normalization stats for Mercator yearly current files.

Expected layout (same as Agulhas training currents)::

  {data_dir}/cur_{year}.nc

with ``uo`` / ``vo`` shaped ``(time, depth, latitude, longitude)``.

Stats are scalar (one mean and one std per variable), aggregated over all
selected timesteps and grid points inside a lon/lat bounding box.

Example::

  python compute_current_norm_stats.py \\
      --data_dir /home/datawork-WW3/FORCING/MERCA_GLOB_PHY_FOR/NC4 \\
      --start_date 20230101 \\
      --end_date 20231231 \\
      --region agulhas \\
      --output mercator_agulhas_2023_stats.json

This JSON is a prep artifact (like ``compute_mfwam_norm_stats.py``). Map
``means`` / ``stds`` for ``uo`` and ``vo`` into
``data.normalization.target`` in the training config.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


NAMED_REGIONS = {
    # lon_min, lon_max, lat_min, lat_max
    "agulhas": (10.0, 35.0, -45.0, -35.0),
    "california_current": (-145.0, -120.0, 30.0, 40.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Mercator regional mean/std for surface currents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing cur_{year}.nc files",
    )
    p.add_argument("--start_date", required=True, help="YYYYMMDD")
    p.add_argument("--end_date", required=True, help="YYYYMMDD (inclusive)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument(
        "--region",
        choices=sorted(NAMED_REGIONS.keys()),
        default=None,
        help="Named bbox (overrides lon/lat args if set)",
    )
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument(
        "--variables",
        nargs="+",
        default=["uo", "vo"],
        help="Current variables to process",
    )
    p.add_argument(
        "--file_pattern",
        default="cur_{year}.nc",
        help="Filename pattern; use {year} (or legacy {YYYY})",
    )
    p.add_argument(
        "--depth_index",
        type=int,
        default=0,
        help="Depth index to keep (surface = 0)",
    )
    p.add_argument(
        "--stride_hours",
        type=int,
        default=1,
        help="Use every N-th hourly step after time filtering (speed vs accuracy)",
    )
    p.add_argument(
        "--time_chunk",
        type=int,
        default=24,
        help="Number of timesteps loaded per batch (memory control)",
    )
    return p.parse_args()


def resolve_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.region is not None:
        return NAMED_REGIONS[args.region]
    missing = [
        name
        for name, val in (
            ("lon_min", args.lon_min),
            ("lon_max", args.lon_max),
            ("lat_min", args.lat_min),
            ("lat_max", args.lat_max),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Missing bbox args {missing}. Pass --region or all lon/lat bounds."
        )
    return args.lon_min, args.lon_max, args.lat_min, args.lat_max


def year_path(data_dir: Path, year: int, pattern: str) -> Path:
    pattern = pattern.replace("{YYYY}", "{year}")
    return data_dir / pattern.format(year=year)


def select_bbox(ds: xr.Dataset, lon_min, lon_max, lat_min, lat_max) -> xr.Dataset:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name]
    lon = ds[lon_name]

    if float(lat[0]) > float(lat[-1]):
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    if lon_min <= lon_max:
        lon_slice = slice(lon_min, lon_max)
        return ds.sel({lat_name: lat_slice, lon_name: lon_slice})

    left = ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, None)})
    right = ds.sel({lat_name: lat_slice, lon_name: slice(None, lon_max)})
    return xr.concat([left, right], dim=lon_name)


class LinearAccumulator:
    """Numerically stable online mean/variance (Welford)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_batch(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        n_b = flat.size
        mean_b = float(flat.mean())
        m2_b = float(((flat - mean_b) ** 2).sum())
        if self.n == 0:
            self.n, self.mean, self.m2 = n_b, mean_b, m2_b
            return
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean = (self.n * self.mean + n_b * mean_b) / n
        self.m2 = self.m2 + m2_b + delta * delta * self.n * n_b / n
        self.n = n

    def result(self) -> dict:
        if self.n < 2:
            std = 0.0
        else:
            std = float(np.sqrt(self.m2 / self.n))  # population std
        return {"mean": float(self.mean), "std": std, "n": int(self.n)}


def _squeeze_depth(ds: xr.Dataset, depth_index: int) -> xr.Dataset:
    if "depth" not in ds.dims:
        return ds
    return ds.isel(depth=depth_index)


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d")
    # Inclusive calendar end of day
    end_inclusive = end + timedelta(days=1) - timedelta(seconds=1)
    lon_min, lon_max, lat_min, lat_max = resolve_bbox(args)
    data_dir = Path(args.data_dir)
    variables = list(args.variables)
    pattern = args.file_pattern.replace("{YYYY}", "{year}")

    years = list(range(start.year, end.year + 1))
    paths = [year_path(data_dir, year, pattern) for year in years]
    existing = [p for p in paths if p.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No files matching {pattern} under {data_dir} for years {years}"
        )

    with xr.open_dataset(existing[0]) as ds0:
        missing = [v for v in variables if v not in ds0.data_vars]
        if missing:
            raise KeyError(f"Variables not in {existing[0].name}: {missing}")

    accumulators = {v: LinearAccumulator() for v in variables}

    print(f"Data dir : {data_dir}")
    print(f"Period   : {start.date()} → {end.date()}")
    print(f"BBox     : lon[{lon_min}, {lon_max}] lat[{lat_min}, {lat_max}]")
    print(f"Vars     : {variables}")
    print(f"Depth idx: {args.depth_index}")
    print(f"Stride h : {args.stride_hours}")
    print(f"Files    : {[p.name for p in existing]}")

    n_files = 0
    n_times = 0
    for path in existing:
        with xr.open_dataset(path, chunks={"time": args.time_chunk}) as ds:
            ds = _squeeze_depth(ds, args.depth_index)
            ds = select_bbox(ds, lon_min, lon_max, lat_min, lat_max)
            # Restrict to requested calendar window (yearly files can overhang).
            ds = ds.sel(time=slice(np.datetime64(start), np.datetime64(end_inclusive)))
            if ds.sizes.get("time", 0) == 0:
                print(f"  skip {path.name}: no times in window")
                continue
            if args.stride_hours > 1:
                ds = ds.isel(time=slice(None, None, args.stride_hours))

            n_t = int(ds.sizes["time"])
            print(f"  {path.name}: {n_t} timesteps after filter/stride")
            for t0 in range(0, n_t, args.time_chunk):
                t1 = min(t0 + args.time_chunk, n_t)
                chunk = ds.isel(time=slice(t0, t1)).load()
                for var in variables:
                    accumulators[var].update_batch(chunk[var].values)
                n_times += t1 - t0
            n_files += 1

    stats = {var: acc.result() for var, acc in accumulators.items()}
    payload = {
        "data_dir": str(data_dir),
        "file_pattern": pattern,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stride_hours": args.stride_hours,
        "depth_index": args.depth_index,
        "bbox": {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "region": args.region,
        },
        "n_files": n_files,
        "n_times": n_times,
        "variables": stats,
        "means": {v: stats[v]["mean"] for v in variables},
        "stds": {v: stats[v]["std"] for v in variables},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Done. Files used={n_files}, timesteps={n_times}")
    print(f"Wrote {out}")
    for v in variables:
        print(
            f"  {v}: mean={stats[v]['mean']:.6g}  "
            f"std={stats[v]['std']:.6g}  n={stats[v]['n']}"
        )


if __name__ == "__main__":
    main()
