"""Generate melt-weighted d_empirical GeoTIFFs matching existing geotiff extents.

For each date's geotiffs in local/debris_shapes/SNFAC/geotiffs/{date}/,
compute d_empirical with HRRR melt weighting and save cropped to the
same spatial extent.

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/generate_melt_d_empirical.py
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from scipy.ndimage import gaussian_filter

from sarvalanche.detection.backscatter_change import calculate_empirical_backscatter_probability
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.weights.temporal import get_temporal_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

NC_DIR = Path("local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns")
GEOTIFF_DIR = Path("local/debris_shapes/SNFAC/geotiffs")
HRRR_PATH = Path("local/issw/hrrr_temperature_sawtooth_2425.nc")
OUT_DIR = Path("local/debris_shapes/SNFAC/geotiffs_melt_filtered")

SEASON_MAP = {
    "2024-2025": "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2023-2024": "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2022-2023": "season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2021-2022": "season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc",
}


def date_to_season(date_str):
    year, month = int(date_str[:4]), int(date_str[5:7])
    return f"{year}-{year+1}" if month >= 9 else f"{year-1}-{year}"


def compute_melt_weighted_d_empirical(ds, date_str, hrrr_ds=None, tau=6):
    """Compute d_empirical with per-pair HRRR melt weighting."""
    ref_ts = np.datetime64(pd.Timestamp(date_str))

    # Clean stale vars
    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    # Compute melt weights per SAR time step
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        sar_times = pd.DatetimeIndex(ds.time.values)
        melt_weights = np.ones(len(sar_times), dtype=np.float32)

        for ti, t in enumerate(sar_times):
            diffs = np.abs(hrrr_times - t)
            ci = diffs.argmin()
            if diffs[ci].days > 2:
                continue
            if 'pdd_24h' in hrrr_ds:
                pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
                pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
                # Scene-mean for per-timestep weight
                w_pdd = max(1.0 - float(np.nanmean(pdd_smooth)) / 0.5, 0.0)
                melt_weights[ti] = min(melt_weights[ti], w_pdd)
            if 't2m_mean' in hrrr_ds:
                t2m = hrrr_ds['t2m_mean'].isel(time=ci).values
                t2m_smooth = gaussian_filter(t2m, sigma=15, mode='nearest')
                w_t2m = min(max((-float(np.nanmean(t2m_smooth)) - 1.0) / 4.0, 0.0), 1.0)
                melt_weights[ti] = min(melt_weights[ti], w_t2m)

        # Apply melt weight to temporal weight
        w_temporal = ds["w_temporal"].values.copy()
        for ti in range(len(sar_times)):
            w_temporal[ti] *= melt_weights[ti]
        ds["w_temporal"].values = w_temporal
        n_melt = (melt_weights < 0.5).sum()
        log.info("  Melt-filtered %d/%d SAR dates", n_melt, len(sar_times))

    p_emp, d_emp = calculate_empirical_backscatter_probability(
        ds, ref_ts, use_agreement_boosting=True,
        agreement_strength=0.8, min_prob_threshold=0.2,
    )
    return d_emp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group dates by season
    date_dirs = sorted(d for d in GEOTIFF_DIR.iterdir() if d.is_dir())
    season_dates = {}
    for dd in date_dirs:
        date_str = dd.name
        season = date_to_season(date_str)
        if season not in season_dates:
            season_dates[season] = []
        season_dates[season].append(date_str)

    log.info("Dates by season: %s", {k: v for k, v in season_dates.items()})

    # Load HRRR (only have 2024-2025 for now)
    hrrr_ds = None
    if HRRR_PATH.exists():
        hrrr_ds = xr.open_dataset(HRRR_PATH)
        log.info("Loaded HRRR from %s", HRRR_PATH)

    for season, dates in sorted(season_dates.items()):
        nc_name = SEASON_MAP.get(season)
        if not nc_name:
            log.warning("No NC mapping for season %s, skipping %s", season, dates)
            continue
        nc_path = NC_DIR / nc_name
        if not nc_path.exists():
            log.warning("NC not found: %s, skipping", nc_path)
            continue

        log.info("Loading %s for season %s (%d dates)", nc_path.name, season, len(dates))
        ds = load_netcdf_to_dataset(nc_path)
        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            ds["time"] = pd.DatetimeIndex(ds["time"].values)
        if any(var.chunks is not None for var in ds.variables.values()):
            ds = ds.load()

        for date_str in dates:
            log.info("Computing d_empirical for %s...", date_str)

            # Check minimum pair span — skip if SAR data has big gaps
            ref_ts = np.datetime64(pd.Timestamp(date_str))
            sar_times = pd.DatetimeIndex(ds.time.values)
            before = sar_times[sar_times <= ref_ts]
            after = sar_times[sar_times > ref_ts]
            if len(before) == 0 or len(after) == 0:
                log.warning("  Skipping %s: no SAR data on one side of date", date_str)
                continue
            min_span = (after[0] - before[-1]).days
            if min_span > 30:
                log.warning("  Skipping %s: min pair span = %d days (>30, unreliable)", date_str, min_span)
                continue

            # Only use HRRR for 2024-2025 season
            use_hrrr = hrrr_ds if season == "2024-2025" else None
            d_emp = compute_melt_weighted_d_empirical(ds, date_str, use_hrrr)

            # Save cropped to each existing geotiff's extent
            gt_dir = GEOTIFF_DIR / date_str
            out_date_dir = OUT_DIR / date_str
            out_date_dir.mkdir(parents=True, exist_ok=True)

            for tif in sorted(gt_dir.glob("*.tif")):
                with rasterio.open(tif) as src:
                    bounds = src.bounds  # left, bottom, right, top

                # Crop d_empirical to these bounds
                try:
                    d_crop = d_emp.rio.clip_box(
                        minx=bounds.left, miny=bounds.bottom,
                        maxx=bounds.right, maxy=bounds.top,
                    )
                except Exception as e:
                    log.warning("  Skipping %s: %s", tif.name, e)
                    continue

                out_name = tif.stem.replace("d_empirical", "d_empirical_melt") + ".tif"
                out_path = out_date_dir / out_name
                d_crop.rio.to_raster(str(out_path))
                log.info("  Saved %s (%dx%d)", out_name, d_crop.sizes.get('x', 0), d_crop.sizes.get('y', 0))

    if hrrr_ds is not None:
        hrrr_ds.close()

    log.info("Done.")


if __name__ == "__main__":
    main()
