"""Evaluate CNN detections against avalanche observations.

For each D2+ observation:
  1. Take its lat/lon point
  2. Run FlowPy from that point to get the expected debris path
  3. Run temporal onset on saved pair probabilities
  4. Check if there's a cluster of 5+ detection pixels within the path
     or within 500m, at time offsets ±1, 3, 7, 10 days
  5. Record detection/miss for each tolerance

Reads saved inference results from run_zone_inference.py.

Usage:
    conda run -n sarvalanche python scripts/evaluation/evaluate_observations.py \
        --obs local/issw/uac/uac_avalanche_observations.csv \
        --inference-dir local/issw/uac/inference/ \
        --nc-dir local/issw/uac/netcdfs/ \
        --center UAC \
        --out eval_results_uac.csv
"""

import argparse
import ast
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.ndimage import binary_dilation, label
from shapely.geometry import Point
from shapely.ops import unary_union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

TIME_TOLERANCES = [1, 3, 7, 10]  # days
MIN_CLUSTER_PIXELS = 5
BUFFER_M = 500  # meters from FlowPy path


# ---------------------------------------------------------------------------
# FlowPy path for a single observation point
# ---------------------------------------------------------------------------

def flowpy_path_for_point(lat, lon, ds, buffer_m=BUFFER_M):
    """Get FlowPy cell_counts mask around an observation point.

    Returns a boolean (y, x) mask of pixels that are either:
    - In a FlowPy runout path (cell_counts > 0) within buffer_m of the point
    - Or directly within buffer_m of the point if no cell_counts available

    Also returns the pixel coordinates (y_idx, x_idx) of the observation.
    """
    crs = ds.rio.crs
    if crs is None:
        raise ValueError("Dataset has no CRS")

    # Project observation point to dataset CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    obs_x, obs_y = transformer.transform(lon, lat)

    # Find nearest pixel
    y_vals = ds.y.values
    x_vals = ds.x.values
    y_idx = int(np.argmin(np.abs(y_vals - obs_y)))
    x_idx = int(np.argmin(np.abs(x_vals - obs_x)))

    # Check bounds
    if y_idx < 0 or y_idx >= len(y_vals) or x_idx < 0 or x_idx >= len(x_vals):
        return None, y_idx, x_idx

    # Compute buffer in pixels
    res = abs(float(x_vals[1] - x_vals[0]))
    buf_px = max(1, int(buffer_m / res))

    H, W = len(y_vals), len(x_vals)
    y_min = max(0, y_idx - buf_px)
    y_max = min(H, y_idx + buf_px + 1)
    x_min = max(0, x_idx - buf_px)
    x_max = min(W, x_idx + buf_px + 1)

    mask = np.zeros((H, W), dtype=bool)

    if "cell_counts" in ds.data_vars:
        cc = ds["cell_counts"].values
        # FlowPy paths near the observation point
        local_cc = cc[y_min:y_max, x_min:x_max]
        local_mask = local_cc > 0

        # Also include a direct circular buffer around the point
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((yy - y_idx)**2 + (xx - x_idx)**2) * res
        local_mask |= dist <= buffer_m

        mask[y_min:y_max, x_min:x_max] = local_mask
    else:
        # No FlowPy — just use circular buffer
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((yy - y_idx)**2 + (xx - x_idx)**2) * res
        mask[y_min:y_max, x_min:x_max] = dist <= buffer_m

    return mask, y_idx, x_idx


# ---------------------------------------------------------------------------
# Temporal onset from saved probabilities
# ---------------------------------------------------------------------------

def load_inference_results(inference_dir):
    """Load pair probabilities and metadata from saved inference."""
    probs_files = sorted(inference_dir.glob("*_pair_probs.npz"))
    meta_files = sorted(inference_dir.glob("*_pair_meta.csv"))

    if not probs_files or not meta_files:
        return None, None

    all_probs = []
    all_metas = []

    for pf, mf in zip(probs_files, meta_files):
        npz = np.load(pf)
        probs = [npz[k] for k in npz.files]
        meta_df = pd.read_csv(mf)
        meta_df['t_start'] = pd.to_datetime(meta_df['t_start'])
        meta_df['t_end'] = pd.to_datetime(meta_df['t_end'])

        metas = meta_df.to_dict('records')
        all_probs.extend(probs)
        all_metas.extend(metas)

    return all_probs, all_metas


def run_onset_cached(inference_dir, onset_cache, threshold=0.2, gap_days=18,
                     hrrr_ds=None, coords=None, crs_str=None):
    """Run temporal onset or load from cache."""
    from sarvalanche.ml.pairwise_debris_classifier.temporal_onset import (
        run_pair_temporal_onset,
    )

    if onset_cache.exists():
        log.info("Loading cached onset from %s", onset_cache)
        result = xr.open_dataset(onset_cache)
        # Load date_fires
        fires_path = onset_cache.with_suffix('.npz')
        if fires_path.exists():
            npz = np.load(fires_path)
            date_fires = npz['date_fires']
            unique_dates = [pd.Timestamp(d) for d in npz['unique_dates']]
        else:
            date_fires = None
            unique_dates = None
        return result, unique_dates, date_fires

    pair_probs, pair_metas = load_inference_results(inference_dir)
    if pair_probs is None:
        return None, None, None

    log.info("Running temporal onset on %d pairs...", len(pair_probs))
    result, unique_dates, date_fires = run_pair_temporal_onset(
        pair_probs, pair_metas,
        threshold=threshold,
        gap_days=gap_days,
        hrrr_ds=hrrr_ds,
        coords=coords,
        crs=crs_str,
    )

    # Cache
    result.to_netcdf(onset_cache)
    np.savez_compressed(onset_cache.with_suffix('.npz'),
                        date_fires=date_fires,
                        unique_dates=[str(d) for d in unique_dates])
    log.info("Cached onset to %s", onset_cache)

    return result, unique_dates, date_fires


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------

def check_detection(date_fires, unique_dates, obs_date, path_mask,
                    time_tol_days, min_pixels=MIN_CLUSTER_PIXELS):
    """Check if a detection cluster exists within the path mask and time window.

    Returns True if there's a connected cluster of >= min_pixels detection
    pixels within path_mask at any date within ±time_tol_days of obs_date.
    """
    if date_fires is None or unique_dates is None:
        return False

    obs_ts = pd.Timestamp(obs_date)

    for di, d in enumerate(unique_dates):
        if abs((d - obs_ts).days) <= time_tol_days:
            fires_at_date = date_fires[di]
            # Pixels that fire AND are in the path
            overlap = fires_at_date & path_mask
            if overlap.sum() < min_pixels:
                continue
            # Check for connected cluster of min_pixels
            labeled, n_features = label(overlap)
            for lbl in range(1, n_features + 1):
                if (labeled == lbl).sum() >= min_pixels:
                    return True

    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_zone(zone_name, nc_path, inference_dir, obs_df,
                  onset_threshold=0.2, onset_gap_days=18):
    """Evaluate all observations for a single zone.

    Returns a DataFrame with one row per observation and detection columns
    for each time tolerance.
    """
    from sarvalanche.io.dataset import load_netcdf_to_dataset

    log.info("Evaluating %s (%d observations)", zone_name, len(obs_df))

    # Load dataset (for CRS, cell_counts, coordinates)
    ds = load_netcdf_to_dataset(nc_path)

    # Get time range of this dataset
    times = pd.DatetimeIndex(ds.time.values)
    ds_start = times.min()
    ds_end = times.max()

    # Filter observations to this dataset's time range (with buffer)
    obs_df = obs_df.copy()
    obs_df['date'] = pd.to_datetime(obs_df['date'])
    max_tol = max(TIME_TOLERANCES)
    obs_in_range = obs_df[
        (obs_df['date'] >= ds_start - pd.Timedelta(days=max_tol)) &
        (obs_df['date'] <= ds_end + pd.Timedelta(days=max_tol))
    ]
    log.info("  %d observations in dataset time range", len(obs_in_range))

    if len(obs_in_range) == 0:
        ds.close()
        return pd.DataFrame()

    # Run temporal onset (cached)
    onset_cache = inference_dir / f"{nc_path.stem}_onset.nc"

    hrrr_ds = None
    if 't2m' in ds.data_vars:
        hrrr_ds = ds[['t2m']]

    result, unique_dates, date_fires = run_onset_cached(
        inference_dir, onset_cache,
        threshold=onset_threshold,
        gap_days=onset_gap_days,
        hrrr_ds=hrrr_ds,
        coords={'y': ds.y.values, 'x': ds.x.values},
        crs_str=str(ds.rio.crs) if ds.rio.crs else None,
    )

    if result is None:
        log.warning("  No inference results for %s", zone_name)
        ds.close()
        return pd.DataFrame()

    # Evaluate each observation
    rows = []
    for _, obs in obs_in_range.iterrows():
        lat, lon = obs.get('lat'), obs.get('lon')
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Get FlowPy path mask for this observation
        path_mask, y_idx, x_idx = flowpy_path_for_point(lat, lon, ds)
        if path_mask is None:
            continue

        row = {
            'date': obs['date'],
            'lat': lat,
            'lon': lon,
            'region': obs.get('region', ''),
            'zone': zone_name,
            'y_idx': y_idx,
            'x_idx': x_idx,
            'path_pixels': int(path_mask.sum()),
        }

        # Check detection at each time tolerance
        for tol in TIME_TOLERANCES:
            row[f'detected_{tol}d'] = check_detection(
                date_fires, unique_dates, obs['date'],
                path_mask, tol, MIN_CLUSTER_PIXELS,
            )

        rows.append(row)

    ds.close()
    log.info("  Evaluated %d observations", len(rows))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN detections against avalanche observations",
    )
    parser.add_argument("--obs", type=Path, required=True,
                        help="Observations CSV (with lat, lon, date columns)")
    parser.add_argument("--inference-dir", type=Path, required=True,
                        help="Directory with saved inference results")
    parser.add_argument("--nc-dir", type=Path, required=True,
                        help="Directory with season netCDFs")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV with detection results")
    parser.add_argument("--onset-threshold", type=float, default=0.2)
    parser.add_argument("--onset-gap-days", type=int, default=18)
    parser.add_argument("--min-d-size", type=float, default=2.0,
                        help="Minimum D-size to include (default: 2.0)")
    args = parser.parse_args()

    # Load observations
    obs_df = pd.read_csv(args.obs)
    log.info("Loaded %d observations", len(obs_df))

    # Parse lat/lon from different formats
    if 'lat' not in obs_df.columns and 'location_point' in obs_df.columns:
        lats, lons = [], []
        for lp in obs_df['location_point']:
            try:
                d = ast.literal_eval(str(lp))
                lats.append(d['lat'])
                lons.append(d['lng'])
            except:
                lats.append(None)
                lons.append(None)
        obs_df['lat'] = lats
        obs_df['lon'] = lons

    # Parse lat/lon from UAC KML-style coordinates
    if 'lat' not in obs_df.columns and 'Coordinates' in obs_df.columns:
        import re
        lats, lons = [], []
        for c in obs_df['Coordinates']:
            m = re.search(r'<coordinates>([-\d.]+),([-\d.]+)', str(c))
            if m:
                lons.append(float(m.group(1)))
                lats.append(float(m.group(2)))
            else:
                lons.append(None)
                lats.append(None)
        obs_df['lat'] = lats
        obs_df['lon'] = lons

    obs_df = obs_df.dropna(subset=['lat', 'lon'])

    # Filter by D-size
    if 'd_size' in obs_df.columns:
        obs_df['d_num'] = pd.to_numeric(obs_df['d_size'], errors='coerce')
        obs_df = obs_df[obs_df['d_num'] >= args.min_d_size]
    elif 'width_ft' in obs_df.columns:
        # UAC uses width/vertical instead of D-size
        w = pd.to_numeric(obs_df['width_ft'], errors='coerce')
        v = pd.to_numeric(obs_df['vertical_ft'], errors='coerce')
        obs_df = obs_df[(w >= 100) | (v >= 300)]

    log.info("After filtering: %d D2+ observations with coordinates", len(obs_df))

    # Find all season NCs and match to inference dirs
    all_results = []
    ncs = sorted(args.nc_dir.rglob("season_*_*.nc"))
    ncs = [nc for nc in ncs if "v2_season" not in str(nc)
           and "v3_" not in str(nc)
           and "probabilities" not in str(nc)]

    for nc in ncs:
        zone_name = nc.parent.name
        inf_dir = args.inference_dir / zone_name

        if not inf_dir.exists() or not list(inf_dir.glob("*_pair_probs.npz")):
            log.info("Skipping %s — no inference results", nc.name)
            continue

        result_df = evaluate_zone(
            zone_name, nc, inf_dir, obs_df,
            onset_threshold=args.onset_threshold,
            onset_gap_days=args.onset_gap_days,
        )
        if not result_df.empty:
            all_results.append(result_df)

    if not all_results:
        log.warning("No results produced")
        return

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(args.out, index=False)
    log.info("Saved %d evaluation rows to %s", len(final), args.out)

    # Summary
    log.info("")
    log.info("=== DETECTION SUMMARY ===")
    total = len(final)
    for tol in TIME_TOLERANCES:
        col = f'detected_{tol}d'
        n_det = final[col].sum()
        rate = n_det / total * 100 if total > 0 else 0
        log.info("  ±%2dd:  %d/%d detected (%.1f%%)", tol, n_det, total, rate)


if __name__ == "__main__":
    main()
