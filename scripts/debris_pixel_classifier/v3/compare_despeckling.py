"""Compare TV denoising strategies: denoise(diff) vs denoise(each) then diff.

Outputs 3 GeoTIFFs for one pair:
  1. Raw dB VV diff (no denoising)
  2. TV denoise the diff (current v3 approach)
  3. TV denoise each image individually, then diff (pipeline approach)

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/compare_despeckling.py \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --out-dir figures/snfac/despeckling_comparison \
        --date 2025-02-04
"""

import argparse
import logging
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from skimage.restoration import denoise_tv_chambolle

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def save_geotiff(data, ds, out_path):
    da = xr.DataArray(
        data, dims=["y", "x"],
        coords={"y": ds.y.values, "x": ds.x.values},
    )
    da = da.rio.set_crs(ds.rio.crs)
    da.rio.to_raster(str(out_path), dtype="float32")
    log.info("  Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True, help="Reference date to find a crossing pair")
    parser.add_argument("--tv-weight", type=float, default=1.0, help="TV weight for all denoising")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_date = pd.Timestamp(args.date)

    # Load
    t0 = _time.time()
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("Loaded in %.1fs", _time.time() - t0)

    # Find a short crossing pair for the reference date
    best_pair = None
    best_span = 999
    for track, pol, da in iter_track_pol_combinations(ds):
        if pol != "VV":
            continue
        if check_db_linear(da) != "dB":
            da = linear_to_dB(da)
        times = pd.DatetimeIndex(da.time.values)
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if not (times[i] <= ref_date < times[j]):
                    continue
                span = (times[j] - times[i]).days
                if span < best_span:
                    best_span = span
                    best_pair = (da, i, j, times, track)

    if best_pair is None:
        log.error("No crossing pair found for %s", ref_date)
        return

    da, i, j, times, track = best_pair
    tag = f"trk{track}_{times[i].date()}_{times[j].date()}_{best_span}d"
    log.info("Using pair: %s", tag)

    img_before = da.isel(time=i).values.astype(np.float32)
    img_after = da.isel(time=j).values.astype(np.float32)

    # 1. Raw diff
    t0 = _time.time()
    raw_diff = img_after - img_before
    raw_diff_clean = np.nan_to_num(raw_diff, nan=0.0)
    log.info("Raw diff computed (%.1fs)", _time.time() - t0)

    # 2. Denoise the diff (v3 approach)
    t0 = _time.time()
    denoised_diff = denoise_tv_chambolle(raw_diff_clean, weight=args.tv_weight).astype(np.float32)
    log.info("Denoise(diff) done (%.1fs, weight=%.1f)", _time.time() - t0, args.tv_weight)

    # 3. Denoise each image, then diff (pipeline approach)
    t0 = _time.time()
    before_filled = np.where(np.isfinite(img_before), img_before, np.nanmean(img_before))
    after_filled = np.where(np.isfinite(img_after), img_after, np.nanmean(img_after))
    before_denoised = denoise_tv_chambolle(before_filled, weight=args.tv_weight).astype(np.float32)
    after_denoised = denoise_tv_chambolle(after_filled, weight=args.tv_weight).astype(np.float32)
    diff_of_denoised = after_denoised - before_denoised
    log.info("Denoise(each) then diff done (%.1fs, weight=%.1f)", _time.time() - t0, args.tv_weight)

    # Mask no-coverage
    valid = np.isfinite(raw_diff) & (np.abs(raw_diff) > 1e-6)
    raw_diff[~valid] = np.nan
    denoised_diff[~valid] = np.nan
    diff_of_denoised[~valid] = np.nan

    # Save GeoTIFFs
    save_geotiff(raw_diff, ds, args.out_dir / f"{tag}_raw_diff.tif")
    save_geotiff(denoised_diff, ds, args.out_dir / f"{tag}_denoise_diff.tif")
    save_geotiff(diff_of_denoised, ds, args.out_dir / f"{tag}_diff_of_denoised.tif")

    # Log stats
    for name, d in [("raw", raw_diff), ("denoise(diff)", denoised_diff), ("diff(denoised)", diff_of_denoised)]:
        v = d[np.isfinite(d)]
        log.info("  %s: mean=%.2f  std=%.2f  >3dB=%d px", name, v.mean(), v.std(), int((v > 3).sum()))

    log.info("Done.")


if __name__ == "__main__":
    main()
