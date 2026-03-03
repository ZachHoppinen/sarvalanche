#!/usr/bin/env python
"""
Export selected variables from a sarvalanche .nc file to GeoTIFFs for QGIS.

Usage
-----
    conda run -n sarvalanche python scripts/nc_to_geotiff.py path/to/run.nc

Outputs GeoTIFFs next to the .nc file:
    run_d_empirical.tif
    run_combined_distance.tif
    run_cell_counts.tif
    run_unmasked_p_target.tif
"""
import argparse
import sys
from pathlib import Path

import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

EXPORT_VARS = [
    'd_empirical',
    'combined_distance',
    # 'cell_counts',
    # 'unmasked_p_target',
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nc_path', type=Path, help='Path to .nc file')
    parser.add_argument('-o', '--outdir', type=Path, default=None,
                        help='Output directory (default: same as .nc file)')
    parser.add_argument('-v', '--vars', nargs='+', default=EXPORT_VARS,
                        help=f'Variables to export (default: {EXPORT_VARS})')
    args = parser.parse_args()

    nc_path = args.nc_path.resolve()
    if not nc_path.exists():
        print(f"ERROR: {nc_path} not found", file=sys.stderr)
        sys.exit(1)

    outdir = (args.outdir or nc_path.parent).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = nc_path.stem

    ds = xr.open_dataset(nc_path)

    # Set CRS from dataset attr if not already set
    crs = ds.attrs.get('crs', None)
    if crs:
        ds = ds.rio.write_crs(crs)

    for var in args.vars:
        if var not in ds.data_vars:
            print(f"  SKIP {var} — not in {nc_path.name}")
            continue

        da = ds[var]
        # rio needs spatial dims set
        da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
        if crs:
            da = da.rio.write_crs(crs)

        out_path = outdir / f'{stem}_{var}.tif'
        da.rio.to_raster(out_path)
        print(f"  {out_path.name}")

    ds.close()
    print(f"Done — {len(args.vars)} vars → {outdir}")


if __name__ == '__main__':
    main()
