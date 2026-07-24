#!/usr/bin/env python3
"""
Standalone downloader for global Mercator surface currents from Copernicus Marine.

Product: GLOBAL_ANALYSISFORECAST_PHY_001_024
  https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description
DOI: https://doi.org/10.48670/moi-00016

Default dataset (6-hourly instantaneous currents, same family used to force MFWAM):
  cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i

Alternative (daily mean currents):
  cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m

Variables: uo (eastward), vo (northward). Surface layer by default (depth ~0–1 m).

Dependencies (pip):
  copernicusmarine
  xarray
  netCDF4

First-time auth (once per machine/user):
  copernicusmarine login

Example:
  python download_surface_currents.py \\
      --start_date 20240101 \\
      --end_date 20240107 \\
      --output_dir ./CUR \\
      --download_dir ./CUR_tmp
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import copernicusmarine
import xarray as xr


PRODUCT_URL = (
    "https://data.marine.copernicus.eu/product/"
    "GLOBAL_ANALYSISFORECAST_PHY_001_024/description"
)

DATASET_IDS = {
    "6hourly": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    "daily": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
}

# CMEMS request name -> optional rename
DEFAULT_VARIABLES = {
    "uo": "u",  # eastward sea water velocity
    "vo": "v",  # northward sea water velocity
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Mercator surface currents (uo, vo) from Copernicus Marine "
            f"product GLOBAL_ANALYSISFORECAST_PHY_001_024. Product page: {PRODUCT_URL}"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start_date", required=True, help="Start date YYYYMMDD")
    parser.add_argument(
        "--end_date",
        default=None,
        help="End date YYYYMMDD (inclusive). Defaults to start_date.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for daily NetCDF files: {output_dir}/{YYYY}/{MM}/{YYYYMMDD}.nc",
    )
    parser.add_argument(
        "--download_dir",
        default=None,
        help="Temporary download directory. Defaults to {output_dir}/_tmp",
    )
    parser.add_argument(
        "--freq",
        choices=sorted(DATASET_IDS.keys()),
        default="6hourly",
        help="Temporal resolution / dataset choice",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES.keys()),
        help="CMEMS variable names to download",
    )
    parser.add_argument(
        "--rename_vars",
        action="store_true",
        help="Rename variables (uo→u, vo→v). Off by default to keep CMEMS names.",
    )
    parser.add_argument(
        "--minimum_depth",
        type=float,
        default=0.0,
        help="Minimum depth (m). Use with --maximum_depth to select the surface layer.",
    )
    parser.add_argument(
        "--maximum_depth",
        type=float,
        default=1.0,
        help="Maximum depth (m). Default keeps only the near-surface level.",
    )
    parser.add_argument(
        "--lon_min",
        type=float,
        default=None,
        help="Optional western longitude bound",
    )
    parser.add_argument(
        "--lon_max",
        type=float,
        default=None,
        help="Optional eastern longitude bound",
    )
    parser.add_argument(
        "--lat_min",
        type=float,
        default=None,
        help="Optional southern latitude bound",
    )
    parser.add_argument(
        "--lat_max",
        type=float,
        default=None,
        help="Optional northern latitude bound",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing daily output files",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary files in download_dir",
    )
    return parser.parse_args()


def date_range(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def output_path_for_day(output_dir: Path, day: datetime) -> Path:
    return (
        output_dir
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.nc"
    )


def download_day(
    day: datetime,
    download_dir: Path,
    dataset_id: str,
    variables: list[str],
    minimum_depth: float,
    maximum_depth: float,
    lon_min: float | None,
    lon_max: float | None,
    lat_min: float | None,
    lat_max: float | None,
) -> Path:
    """Download one calendar day of surface currents into download_dir."""
    tmp_name = f"cur_{day.strftime('%Y%m%d')}.nc"
    tmp_path = download_dir / tmp_name

    kwargs = {
        "dataset_id": dataset_id,
        "start_datetime": f"{day.strftime('%Y-%m-%d')}T00:00:00",
        "end_datetime": f"{day.strftime('%Y-%m-%d')}T23:59:59",
        "variables": variables,
        "minimum_depth": minimum_depth,
        "maximum_depth": maximum_depth,
        "output_filename": tmp_name,
        "output_directory": str(download_dir),
        "overwrite": True,
    }
    if lon_min is not None:
        kwargs["minimum_longitude"] = lon_min
    if lon_max is not None:
        kwargs["maximum_longitude"] = lon_max
    if lat_min is not None:
        kwargs["minimum_latitude"] = lat_min
    if lat_max is not None:
        kwargs["maximum_latitude"] = lat_max

    print(f"Downloading {day.strftime('%Y-%m-%d')} ...")
    copernicusmarine.subset(**kwargs)

    if not tmp_path.exists():
        raise FileNotFoundError(f"Expected download missing: {tmp_path}")
    return tmp_path


def postprocess_and_save(
    tmp_path: Path,
    out_path: Path,
    rename_vars: bool,
    variables: list[str],
) -> None:
    """Optionally rename variables, squeeze singleton depth, write daily NetCDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(tmp_path) as ds:
        # Drop singleton depth so files are 2D time/lat/lon surface fields
        for depth_name in ("depth", "deptht"):
            if depth_name in ds.dims and ds.sizes.get(depth_name, 0) == 1:
                ds = ds.squeeze(depth_name, drop=True)

        if rename_vars:
            rename_map = {
                src: DEFAULT_VARIABLES[src]
                for src in variables
                if src in DEFAULT_VARIABLES
                and src in ds.data_vars
                and DEFAULT_VARIABLES[src] != src
            }
            if rename_map:
                ds = ds.rename(rename_map)
                print(f"  Renamed: {rename_map}")

        encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in ds.data_vars
        }
        ds.to_netcdf(out_path, encoding=encoding)

    print(f"  Saved {out_path}")


def main() -> None:
    args = parse_args()

    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = (
        datetime.strptime(args.end_date, "%Y%m%d")
        if args.end_date
        else start
    )
    if end < start:
        raise ValueError("end_date must be >= start_date")

    dataset_id = DATASET_IDS[args.freq]
    output_dir = Path(args.output_dir).expanduser().resolve()
    download_dir = (
        Path(args.download_dir).expanduser().resolve()
        if args.download_dir
        else output_dir / "_tmp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {dataset_id} (freq={args.freq})")
    print(f"Product : {PRODUCT_URL}")
    print(f"Period  : {start.date()} → {end.date()}")
    print(f"Depth   : {args.minimum_depth} → {args.maximum_depth} m")
    print(f"Output  : {output_dir}")
    print(f"Tmp     : {download_dir}")
    print(f"Vars    : {args.variables}")
    print(f"Rename  : {args.rename_vars}")

    for day in date_range(start, end):
        out_path = output_path_for_day(output_dir, day)
        if out_path.exists() and not args.overwrite:
            print(f"Skipping {day.strftime('%Y%m%d')} (already exists)")
            continue

        tmp_path = download_day(
            day=day,
            download_dir=download_dir,
            dataset_id=dataset_id,
            variables=args.variables,
            minimum_depth=args.minimum_depth,
            maximum_depth=args.maximum_depth,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
        )
        postprocess_and_save(
            tmp_path=tmp_path,
            out_path=out_path,
            rename_vars=args.rename_vars,
            variables=args.variables,
        )
        if not args.keep_tmp:
            tmp_path.unlink(missing_ok=True)

    if not args.keep_tmp and download_dir.exists():
        try:
            if not any(download_dir.iterdir()):
                download_dir.rmdir()
        except OSError:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
