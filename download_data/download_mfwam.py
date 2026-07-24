#!/usr/bin/env python3
"""
Standalone downloader for global MFWAM analysis & forecast waves from Copernicus Marine.

Product: GLOBAL_ANALYSISFORECAST_WAV_001_027
  https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_WAV_001_027/description
Dataset ID: cmems_mod_glo_wav_anfc_0.083deg_PT3H-i  (~1/12°, 3-hourly)

Dependencies (pip):
  copernicusmarine
  xarray
  netCDF4

First-time auth (once per machine/user):
  copernicusmarine login

Example:
  python download_mfwam.py \\
      --start_date 20240101 \\
      --end_date 20240107 \\
      --output_dir ./MFWAM \\
      --download_dir ./MFWAM_tmp
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import copernicusmarine
import xarray as xr


DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
PRODUCT_URL = (
    "https://data.marine.copernicus.eu/product/"
    "GLOBAL_ANALYSISFORECAST_WAV_001_027/description"
)

# CMEMS request name -> optional Amphitrite-style rename
DEFAULT_VARIABLES = {
    "VHM0": "swh",      # significant wave height
    "VMDR": "mwd",      # mean wave direction
    "VTM10": "mwp",     # mean period Tm-10
    "VTPK": "pp1d",     # peak period
    "VHM0_WW": "VHM0_WW",
    "VTM01_WW": "VTM01_WW",
    "VMDR_WW": "VMDR_WW",
    "VHM0_SW1": "VHM0_SW1",
    "VTM01_SW1": "VTM01_SW1",
    "VMDR_SW1": "VMDR_SW1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download global MFWAM wave fields from Copernicus Marine "
            f"({DATASET_ID}). Product page: {PRODUCT_URL}"
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
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES.keys()),
        help="CMEMS variable names to download",
    )
    parser.add_argument(
        "--rename_vars",
        action="store_true",
        help="Rename variables (e.g. VHM0→swh). Off by default to keep CMEMS names.",
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
    variables: list[str],
    lon_min: float | None,
    lon_max: float | None,
    lat_min: float | None,
    lat_max: float | None,
) -> Path:
    """Download one calendar day of MFWAM fields into download_dir."""
    tmp_name = f"mfwam_{day.strftime('%Y%m%d')}.nc"
    tmp_path = download_dir / tmp_name

    kwargs = {
        "dataset_id": DATASET_ID,
        "start_datetime": f"{day.strftime('%Y-%m-%d')}T00:00:00",
        "end_datetime": f"{day.strftime('%Y-%m-%d')}T23:59:59",
        "variables": variables,
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
    """Optionally rename variables and write the final daily NetCDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(tmp_path) as ds:
        if rename_vars:
            rename_map = {
                src: DEFAULT_VARIABLES[src]
                for src in variables
                if src in DEFAULT_VARIABLES and src in ds.data_vars
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

    output_dir = Path(args.output_dir).expanduser().resolve()
    download_dir = (
        Path(args.download_dir).expanduser().resolve()
        if args.download_dir
        else output_dir / "_tmp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {DATASET_ID}")
    print(f"Product : {PRODUCT_URL}")
    print(f"Period  : {start.date()} → {end.date()}")
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
            variables=args.variables,
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
        # Remove empty tmp dir when we created the default one under output_dir
        try:
            if not any(download_dir.iterdir()):
                download_dir.rmdir()
        except OSError:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
