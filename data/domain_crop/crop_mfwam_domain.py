#!/usr/bin/env python3
"""Crop an MFWAM NetCDF file to a longitude/latitude bounding box.

Example:
    python data/domain_crop/crop_mfwam_domain.py \
        --input data/raw/mfwam_smoke/2023/01/20230101.nc \
        --output data/domain_crop/agulhas_20230101.nc \
        --lon-min 10 --lon-max 35 --lat-min -45 --lat-max -35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop an MFWAM NetCDF file to a longitude/latitude bounding box."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input NetCDF file")
    parser.add_argument("--output", required=True, type=Path, help="Output NetCDF file")
    parser.add_argument("--lon-min", required=True, type=float, help="Western bound")
    parser.add_argument("--lon-max", required=True, type=float, help="Eastern bound")
    parser.add_argument("--lat-min", required=True, type=float, help="Southern bound")
    parser.add_argument("--lat-max", required=True, type=float, help="Northern bound")
    return parser.parse_args()


def coordinate_slice(coordinate: xr.DataArray, minimum: float, maximum: float) -> slice:
    if minimum > maximum:
        raise ValueError("Bounding box must not cross the antimeridian")
    if coordinate.size < 2:
        raise ValueError(f"Coordinate {coordinate.name!r} must contain at least two values")
    if float(coordinate[0]) <= float(coordinate[-1]):
        return slice(minimum, maximum)
    return slice(maximum, minimum)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input NetCDF does not exist: {args.input}")

    with xr.open_dataset(args.input) as dataset:
        latitude_name = "latitude" if "latitude" in dataset.coords else "lat"
        longitude_name = "longitude" if "longitude" in dataset.coords else "lon"
        if latitude_name not in dataset.coords or longitude_name not in dataset.coords:
            raise KeyError("Dataset must provide latitude/longitude or lat/lon coordinates")

        cropped = dataset.sel(
            {
                latitude_name: coordinate_slice(
                    dataset[latitude_name], args.lat_min, args.lat_max
                ),
                longitude_name: coordinate_slice(
                    dataset[longitude_name], args.lon_min, args.lon_max
                ),
            }
        )
        if cropped.sizes[latitude_name] == 0 or cropped.sizes[longitude_name] == 0:
            raise ValueError("Bounding box does not overlap the input dataset")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            variable: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for variable in cropped.data_vars
        }
        cropped.to_netcdf(args.output, encoding=encoding)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(
        f"BBox  : lon[{args.lon_min}, {args.lon_max}] "
        f"lat[{args.lat_min}, {args.lat_max}]"
    )


if __name__ == "__main__":
    main()
