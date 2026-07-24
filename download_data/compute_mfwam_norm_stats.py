#!/usr/bin/env python3
"""
Compute regional mean / std normalization stats for MFWAM daily NetCDF files.

Expected layout (same as Amphitrite BASELINES):
  {data_dir}/{YYYY}/{MM}/{YYYYMMDD}.nc

Stats are scalar (one mean and one std per variable), aggregated over all
timesteps and grid points inside a lon/lat bounding box.

Example:
  python compute_mfwam_norm_stats.py \\
      --data_dir /home/datawork-WW3/PROJECT/AMPHITRITE/BASELINES/MFWAM \\
      --start_date 20230101 \\
      --end_date 20231231 \\
      --lon_min 10 --lon_max 50 --lat_min -45 --lat_max -25 \\
      --output mfwam_agulhas_2023_stats.json

Direction variables (VMDR*) should use --circular_vars for circular mean/std.
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
    "agulhas": (10.0, 50.0, -45.0, -25.0),
    "gulfstream": (-80.0, -40.0, 25.0, 45.0),
    "biscay": (-10.0, 0.0, 43.0, 50.0),
    "california_current": (-145.0, -120.0, 30.0, 40.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute MFWAM regional mean/std per variable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Root of MFWAM tree (contains YYYY/MM/YYYYMMDD.nc)",
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
        default=None,
        help="Variables to process (default: all data_vars in first file)",
    )
    p.add_argument(
        "--circular_vars",
        nargs="*",
        default=["VMDR", "VMDR_WW", "VMDR_SW1", "VMDR_SW2", "mwd"],
        help="Direction variables for which circular mean/std are used",
    )
    p.add_argument(
        "--stride_days",
        type=int,
        default=1,
        help="Use every N-th day (speed vs accuracy trade-off)",
    )
    return p.parse_args()


def date_range(start: datetime, end: datetime, stride: int):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=stride)


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


def daily_path(data_dir: Path, day: datetime) -> Path:
    return (
        data_dir
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.nc"
    )


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

    # Antimeridian wrap: lon_min > lon_max
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
        """Faster batch update via parallel Welford merge of a chunk."""
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


class CircularAccumulator:
    """Circular mean / std for directions in degrees (0–360)."""

    def __init__(self):
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.n = 0

    def update_batch(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        rad = np.deg2rad(flat)
        self.sum_cos += float(np.cos(rad).sum())
        self.sum_sin += float(np.sin(rad).sum())
        self.n += flat.size

    def result(self) -> dict:
        if self.n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        mean_rad = np.arctan2(self.sum_sin / self.n, self.sum_cos / self.n)
        r = np.hypot(self.sum_cos / self.n, self.sum_sin / self.n)
        r = min(1.0, max(0.0, float(r)))
        std_rad = np.sqrt(max(0.0, -2.0 * np.log(r))) if r > 0 else np.pi
        return {
            "mean": float(np.rad2deg(mean_rad) % 360.0),
            "std": float(np.rad2deg(std_rad)),
            "n": int(self.n),
        }


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d")
    lon_min, lon_max, lat_min, lat_max = resolve_bbox(args)
    data_dir = Path(args.data_dir)
    circular = set(args.circular_vars or [])

    # Discover variables from first existing file if needed
    variables = args.variables
    sample = None
    for day in date_range(start, end, args.stride_days):
        path = daily_path(data_dir, day)
        if path.is_file():
            sample = path
            break
    if sample is None:
        raise FileNotFoundError(f"No NetCDF found under {data_dir} for the period")

    with xr.open_dataset(sample) as ds0:
        if variables is None:
            variables = [v for v in ds0.data_vars]
        else:
            missing = [v for v in variables if v not in ds0.data_vars]
            if missing:
                raise KeyError(f"Variables not in {sample.name}: {missing}")

    accumulators = {
        v: (CircularAccumulator() if v in circular else LinearAccumulator())
        for v in variables
    }

    print(f"Data dir : {data_dir}")
    print(f"Period   : {start.date()} → {end.date()} (stride={args.stride_days})")
    print(f"BBox     : lon[{lon_min}, {lon_max}] lat[{lat_min}, {lat_max}]")
    print(f"Vars     : {variables}")
    print(f"Circular : {sorted(circular & set(variables))}")

    n_files = 0
    n_missing = 0
    for day in date_range(start, end, args.stride_days):
        path = daily_path(data_dir, day)
        if not path.is_file():
            n_missing += 1
            continue
        with xr.open_dataset(path) as ds:
            ds_r = select_bbox(ds, lon_min, lon_max, lat_min, lat_max)
            for var in variables:
                accumulators[var].update_batch(ds_r[var].values)
        n_files += 1
        if n_files % 10 == 0:
            print(f"  processed {n_files} files (last {path.name})")

    stats = {var: acc.result() for var, acc in accumulators.items()}
    payload = {
        "data_dir": str(data_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stride_days": args.stride_days,
        "bbox": {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "region": args.region,
        },
        "n_files": n_files,
        "n_missing": n_missing,
        "circular_vars": sorted(circular & set(variables)),
        "variables": stats,
        # Convenience flat dicts often used by training configs
        "means": {v: stats[v]["mean"] for v in variables},
        "stds": {v: stats[v]["std"] for v in variables},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Done. Files used={n_files}, missing={n_missing}")
    print(f"Wrote {out}")
    for v in variables:
        print(f"  {v}: mean={stats[v]['mean']:.6g}  std={stats[v]['std']:.6g}  n={stats[v]['n']}")


if __name__ == "__main__":
    main()
