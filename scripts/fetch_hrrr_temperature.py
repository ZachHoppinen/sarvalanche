"""Fetch HRRR-Alaska 2m temperature for SAR dates and lapse-rate adjust to DEM.

Downloads TMP:2m from HRRR-AK (3km) for each SAR date, computes daily max/mean,
resamples to match the SAR grid, and applies elevation-based lapse rate correction
using the high-res DEM.

Output: NetCDF with per-date t2m_max, t2m_mean, above_freezing_hours variables
aligned to the SAR grid.

Usage:
    conda run -n sarvalanche python scripts/fetch_hrrr_temperature.py \
        --nc local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_*.nc \
        --out local/cnfaic/hrrr_temperature_2025-2026.nc

    # Both seasons:
    conda run -n sarvalanche python scripts/fetch_hrrr_temperature.py \
        --nc local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_*.nc \
            local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2024-2025_*.nc \
        --out local/cnfaic/hrrr_temperature.nc
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from herbie import Herbie
from scipy.interpolate import RegularGridInterpolator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

LAPSE_RATE = -6.5e-3  # °C per meter (standard atmosphere)
HOURS_TO_FETCH = [0, 3, 6, 9, 12, 15, 18, 21]  # UTC hours to sample
KELVIN_TO_C = -273.15


def get_hrrr_t2m(date_str, hour):
    """Fetch HRRR-AK 2m temperature for one date/hour. Returns (t2m_K, lat, lon) or None."""
    try:
        H = Herbie(f"{date_str} {hour:02d}:00", model="hrrrak", product="sfc", fxx=0, verbose=False)
        ds = H.xarray("TMP:2 m above ground")
        t2m = ds["t2m"].values  # (y, x) in Kelvin
        lat = ds["latitude"].values  # (y, x)
        lon = ds["longitude"].values  # (y, x) — 0-360 convention
        ds.close()
        return t2m, lat, lon
    except Exception as e:
        log.debug("  HRRR unavailable for %s %02d:00: %s", date_str, hour, e)
        return None


def resample_to_sar_grid(t2m_K, hrrr_lat, hrrr_lon, sar_lat, sar_lon):
    """Resample HRRR grid (3km) to SAR grid (30m) using bilinear interpolation.

    HRRR uses 0-360 longitude; SAR uses -180 to 180. Convert.
    """
    # Convert HRRR lon from 0-360 to -180/180
    hrrr_lon_180 = np.where(hrrr_lon > 180, hrrr_lon - 360, hrrr_lon)

    # Build interpolator from HRRR curvilinear grid
    # Since HRRR is curvilinear, use nearest-neighbor via flattened approach
    from scipy.spatial import cKDTree

    # Flatten HRRR coords
    flat_lat = hrrr_lat.ravel()
    flat_lon = hrrr_lon_180.ravel()
    flat_t2m = t2m_K.ravel()

    # Build KDTree on HRRR lat/lon
    tree = cKDTree(np.column_stack([flat_lat, flat_lon]))

    # Query for each SAR pixel
    sar_points = np.column_stack([sar_lat.ravel(), sar_lon.ravel()])
    dist, idx = tree.query(sar_points)
    result = flat_t2m[idx].reshape(sar_lat.shape)

    return result


def process_date(date_str, sar_lat, sar_lon, dem, hours=None):
    """Fetch HRRR for multiple hours, compute daily stats, lapse-rate adjust.

    Returns dict with t2m_max, t2m_mean, above_freezing_hours (all in °C, adjusted to DEM).
    """
    if hours is None:
        hours = HOURS_TO_FETCH
    hourly_grids = []

    for hour in hours:
        result = get_hrrr_t2m(date_str, hour)
        if result is None:
            continue
        t2m_K, hrrr_lat, hrrr_lon = result

        # Resample to SAR grid
        t2m_sar = resample_to_sar_grid(t2m_K, hrrr_lat, hrrr_lon, sar_lat, sar_lon)

        # Convert to Celsius
        t2m_C = t2m_sar + KELVIN_TO_C

        hourly_grids.append(t2m_C)

    if not hourly_grids:
        log.warning("  No HRRR data for %s", date_str)
        return None

    hourly = np.stack(hourly_grids, axis=0)  # (n_hours, y, x)
    log.info("  %s: %d hours fetched", date_str, len(hourly_grids))

    # HRRR reports t2m at its own grid elevation. Adjust to our DEM.
    # HRRR-AK terrain is ~3km smoothed. We need the HRRR terrain elevation
    # to compute the correction. For simplicity, use the mean HRRR t2m
    # and apply lapse rate relative to a reference elevation.
    # Since we're comparing relative (above/below freezing), the lapse rate
    # correction from HRRR grid elevation (~smoothed) to actual DEM matters.
    # Use: T_adjusted = T_hrrr + LAPSE_RATE * (dem - dem_smoothed)
    # Approximate dem_smoothed by smoothing our DEM to ~3km
    from scipy.ndimage import uniform_filter
    # Estimate pixel size from DEM shape vs typical extent
    # At 30m resolution, 3km ≈ 100 pixels
    smooth_size = 100
    dem_smoothed = uniform_filter(dem, size=smooth_size)
    elev_diff = dem - dem_smoothed  # positive = pixel is higher than HRRR grid

    # Apply lapse rate to each hourly grid
    lapse_correction = LAPSE_RATE * elev_diff  # °C
    hourly_adjusted = hourly + lapse_correction[np.newaxis, :, :]

    t2m_max = np.nanmax(hourly_adjusted, axis=0)
    t2m_mean = np.nanmean(hourly_adjusted, axis=0)
    above_freezing_hours = (hourly_adjusted > 0).sum(axis=0).astype(np.float32)

    # Cumulative positive degree-day (PDD) over the sampled hours.
    # Sum positive temperatures, scale by (24 / n_samples) to approximate
    # the integral over 24 hours in degree-day units.
    n_samples = len(hourly_grids)
    positive_temps = np.clip(hourly_adjusted, 0, None)
    # Each sample represents ~(24/n_samples) hours
    hours_per_sample = 24.0 / n_samples
    pdd_24h = np.nansum(positive_temps, axis=0) * hours_per_sample / 24.0  # degree-days

    return {
        "t2m_max": t2m_max.astype(np.float32),
        "t2m_mean": t2m_mean.astype(np.float32),
        "above_freezing_hours": above_freezing_hours,
        "pdd_24h": pdd_24h.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch HRRR-AK 2m temperature for SAR dates")
    parser.add_argument("--nc", type=Path, required=True, nargs="+",
                        help="Season NetCDF file(s)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output NetCDF path")
    parser.add_argument("--hours", type=int, nargs="+", default=HOURS_TO_FETCH,
                        help="UTC hours to fetch (default: 0 3 6 9 12 15 18 21)")
    args = parser.parse_args()

    hours_to_fetch = args.hours

    # Collect unique SAR dates from all input files
    all_dates = set()
    ref_ds = None
    for nc_path in args.nc:
        ds = xr.open_dataset(nc_path)
        times = pd.DatetimeIndex(ds.time.values)
        for t in times:
            all_dates.add(t.strftime("%Y-%m-%d"))
        if ref_ds is None:
            ref_ds = ds  # keep first one open for grid info
        else:
            ds.close()

    all_dates = sorted(all_dates)
    log.info("Processing %d unique SAR dates from %d files", len(all_dates), len(args.nc))

    # Build lat/lon grids for SAR
    # The dataset is in EPSG:4326, so x=lon, y=lat
    sar_lon_1d = ref_ds.x.values
    sar_lat_1d = ref_ds.y.values
    sar_lon, sar_lat = np.meshgrid(sar_lon_1d, sar_lat_1d)

    # DEM for lapse rate correction
    if "dem" in ref_ds:
        dem = ref_ds["dem"].values
        if dem.ndim == 3:
            dem = dem[0]  # take first time step if 3D
        # Fill NaN with 0 (water/ocean)
        dem = np.where(np.isfinite(dem), dem, 0)
    else:
        log.warning("No DEM in dataset, skipping lapse rate correction")
        dem = np.zeros_like(sar_lat)

    # Process each date
    results = {}
    for date_str in all_dates:
        log.info("Date: %s", date_str)
        r = process_date(date_str, sar_lat, sar_lon, dem, hours=hours_to_fetch)
        if r is not None:
            results[date_str] = r

    ref_ds.close()

    if not results:
        log.error("No data retrieved")
        return

    # Build output dataset
    dates_with_data = sorted(results.keys())
    n_dates = len(dates_with_data)
    ny, nx = sar_lat.shape

    t2m_max = np.full((n_dates, ny, nx), np.nan, dtype=np.float32)
    t2m_mean = np.full((n_dates, ny, nx), np.nan, dtype=np.float32)
    above_freezing = np.full((n_dates, ny, nx), np.nan, dtype=np.float32)
    pdd_24h = np.full((n_dates, ny, nx), np.nan, dtype=np.float32)

    for i, d in enumerate(dates_with_data):
        t2m_max[i] = results[d]["t2m_max"]
        t2m_mean[i] = results[d]["t2m_mean"]
        above_freezing[i] = results[d]["above_freezing_hours"]
        pdd_24h[i] = results[d]["pdd_24h"]

    time_coord = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_with_data])

    out_ds = xr.Dataset(
        {
            "t2m_max": (["time", "y", "x"], t2m_max, {
                "units": "degC", "long_name": "Daily max 2m temperature (lapse-rate adjusted)",
            }),
            "t2m_mean": (["time", "y", "x"], t2m_mean, {
                "units": "degC", "long_name": "Daily mean 2m temperature (lapse-rate adjusted)",
            }),
            "above_freezing_hours": (["time", "y", "x"], above_freezing, {
                "units": "count", "long_name": "Number of sampled hours with T > 0°C",
            }),
            "pdd_24h": (["time", "y", "x"], pdd_24h, {
                "units": "degC*day", "long_name": "24h cumulative positive degree-day (lapse-rate adjusted)",
            }),
        },
        coords={
            "time": time_coord,
            "y": sar_lat_1d,
            "x": sar_lon_1d,
        },
    )
    out_ds.attrs["source"] = "HRRR-Alaska (3km) via Herbie, lapse-rate adjusted"
    out_ds.attrs["lapse_rate"] = f"{LAPSE_RATE} C/m"
    out_ds.attrs["hours_sampled"] = str(HOURS_TO_FETCH)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(args.out)
    log.info("Saved %s (%d dates, %.1f MB)",
             args.out, n_dates, args.out.stat().st_size / 1e6)

    # Print summary
    for i, d in enumerate(dates_with_data):
        tmax = float(np.nanmax(t2m_max[i]))
        tmean = float(np.nanmean(t2m_mean[i]))
        n_above = float(np.nanmean(above_freezing[i]))
        log.info("  %s  Tmax=%.1f°C  Tmean=%.1f°C  above_freezing=%.1f hrs (mean)",
                 d, tmax, tmean, n_above)


if __name__ == "__main__":
    main()
