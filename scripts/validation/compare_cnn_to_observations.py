#!/usr/bin/env python3
"""Compare CNN debris probabilities against FlowPy observation paths.

Two comparison approaches:
  1. Zonal statistics — mean/max CNN probability inside each FlowPy path
     vs. a background annulus, at the nearest CNN time step.
  2. Onset date comparison — difference between CNN-detected onset date
     and the reported observation date.

Outputs a summary CSV to local/issw/observations/comparison_summary.csv.

Usage:
    conda run -n sarvalanche python scripts/validation/compare_cnn_to_observations.py
    conda run -n sarvalanche python scripts/validation/compare_cnn_to_observations.py --max-time-gap 12
"""

import argparse
import ast
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from shapely.geometry import box
from shapely.ops import unary_union

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

ROOT = Path(__file__).resolve().parents[2]
OBS_DIR = ROOT / "local/issw/observations"
NC_DIR = ROOT / "local/issw/netcdfs"
CSV_PATH = ROOT / "local/issw/snfac_obs_2021_2025.csv"
OUT_CSV = OBS_DIR / "comparison_summary.csv"

# CNN data sources: (zone_dir, has_seasonal_nc, has_temporal_onset, scene_tif_dir)
CNN_SOURCES = [
    {
        "name": "Banner_Summit",
        "seasonal_nc": NC_DIR / "Banner_Summit/v2_season_inference/season_v2_debris_probabilities.nc",
        "temporal_onset_nc": NC_DIR / "Banner_Summit/v2_season_inference/temporal_onset.nc",
        "scene_tif_dir": None,
    },
    {
        "name": "Galena_Summit_&_Eastern_Mtns",
        "seasonal_nc": None,
        "temporal_onset_nc": None,
        "scene_tif_dir": NC_DIR / "Galena_Summit_&_Eastern_Mtns/v2_inference",
    },
    {
        "name": "Sawtooth_&_Western_Smoky_Mtns",
        "seasonal_nc": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns/v2_season_inference/season_v2_debris_probabilities.nc",
        "temporal_onset_nc": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns/v2_season_inference/temporal_onset.nc",
        "scene_tif_dir": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns/v2_season_inference",
    },
]

# Spatial buffer around FlowPy paths (metres, applied in UTM before reprojecting)
PATH_BUFFER_M = 500
# Background annulus extends this far beyond the buffered path (degrees, ~500m)
BACKGROUND_BUFFER_DEG = 0.005


def parse_location_point(loc_str: str) -> tuple[float, float]:
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def load_observations_df() -> pd.DataFrame:
    """Load CSV and parse lat/lng."""
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["location_point", "date"])
    lats, lngs = [], []
    for loc in df["location_point"]:
        lat, lng = parse_location_point(loc)
        lats.append(lat)
        lngs.append(lng)
    df["lat"] = lats
    df["lng"] = lngs
    df["date"] = pd.to_datetime(df["date"])
    return df


def rasterize_path_to_grid(
    path_geom, da: xr.DataArray
) -> np.ndarray:
    """Burn a shapely geometry into a boolean mask on the DataArray's grid."""
    from rasterio.transform import from_bounds

    ny, nx = da.sizes["y"], da.sizes["x"]
    bounds = da.rio.bounds()  # (left, bottom, right, top)
    transform = from_bounds(*bounds, nx, ny)
    mask = rasterio.features.geometry_mask(
        [path_geom],
        out_shape=(ny, nx),
        transform=transform,
        invert=True,  # True inside geometry
    )
    return mask


def zonal_stats_for_path(
    prob_2d: xr.DataArray,
    path_geom,
) -> dict:
    """Compute CNN probability stats inside the path and in a background annulus."""
    path_mask = rasterize_path_to_grid(path_geom, prob_2d)
    prob_vals = prob_2d.values

    # Stats inside path
    inside = prob_vals[path_mask]
    inside = inside[np.isfinite(inside)]

    if len(inside) == 0:
        return {
            "n_pixels_path": 0,
            "max_prob_path": np.nan,
            "mean_prob_path": np.nan,
            "p90_prob_path": np.nan,
            "n_pixels_bg": 0,
            "mean_prob_bg": np.nan,
            "max_prob_bg": np.nan,
        }

    # Background: buffer the path, then exclude the path itself
    bg_geom = path_geom.buffer(BACKGROUND_BUFFER_DEG).difference(path_geom)
    bg_mask = rasterize_path_to_grid(bg_geom, prob_2d)
    outside = prob_vals[bg_mask]
    outside = outside[np.isfinite(outside)]

    return {
        "n_pixels_path": len(inside),
        "max_prob_path": float(np.max(inside)),
        "mean_prob_path": float(np.mean(inside)),
        "p90_prob_path": float(np.percentile(inside, 90)),
        "n_pixels_bg": len(outside),
        "mean_prob_bg": float(np.mean(outside)) if len(outside) > 0 else np.nan,
        "max_prob_bg": float(np.max(outside)) if len(outside) > 0 else np.nan,
    }


def onset_stats_for_path(
    onset_ds: xr.Dataset,
    path_geom,
    obs_date: pd.Timestamp,
) -> dict:
    """Compare CNN onset dates inside path to observation date."""
    onset_da = onset_ds["onset_date"]
    candidate_da = onset_ds["candidate_mask"]
    confidence_da = onset_ds["confidence"]
    peak_prob_da = onset_ds["peak_prob"]

    path_mask = rasterize_path_to_grid(path_geom, onset_da)

    # Only look at candidate pixels inside path
    combined = path_mask & candidate_da.values.astype(bool)
    n_candidates = int(combined.sum())

    if n_candidates == 0:
        return {
            "n_candidates_in_path": 0,
            "onset_date_diff_days_mean": np.nan,
            "onset_date_diff_days_median": np.nan,
            "onset_date_diff_days_min": np.nan,
            "mean_confidence": np.nan,
            "mean_peak_prob": np.nan,
        }

    onset_dates = onset_da.values[combined]
    valid = ~np.isnat(onset_dates)
    onset_dates = onset_dates[valid]

    if len(onset_dates) == 0:
        return {
            "n_candidates_in_path": n_candidates,
            "onset_date_diff_days_mean": np.nan,
            "onset_date_diff_days_median": np.nan,
            "onset_date_diff_days_min": np.nan,
            "mean_confidence": np.nan,
            "mean_peak_prob": np.nan,
        }

    obs_dt64 = np.datetime64(obs_date)
    diffs_days = (onset_dates - obs_dt64) / np.timedelta64(1, "D")

    conf_vals = confidence_da.values[combined]
    peak_vals = peak_prob_da.values[combined]

    return {
        "n_candidates_in_path": n_candidates,
        "onset_date_diff_days_mean": float(np.nanmean(diffs_days)),
        "onset_date_diff_days_median": float(np.nanmedian(diffs_days)),
        "onset_date_diff_days_min": float(np.min(np.abs(diffs_days))),
        "mean_confidence": float(np.nanmean(conf_vals)),
        "mean_peak_prob": float(np.nanmean(peak_vals)),
    }


def find_nearest_time_idx(times: np.ndarray, target: pd.Timestamp) -> int:
    """Return index of nearest time step."""
    diffs = np.abs(times - np.datetime64(target))
    return int(np.argmin(diffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-time-gap", type=float, default=None,
        help="Only compare observations within N days of a CNN time step.",
    )
    parser.add_argument(
        "--path-buffer-m", type=float, default=PATH_BUFFER_M,
        help="Spatial buffer (metres) to add around FlowPy paths (default 500).",
    )
    args = parser.parse_args()

    obs_df = load_observations_df()
    all_paths = gpd.read_file(OBS_DIR / "all_flowpy_paths.gpkg")

    # Buffer paths in their native UTM CRS, then reproject to EPSG:4326
    if args.path_buffer_m > 0:
        log.info("Buffering FlowPy paths by %.0f m (in %s)", args.path_buffer_m, all_paths.crs)
        all_paths["geometry"] = all_paths.geometry.buffer(args.path_buffer_m)
    if all_paths.crs and all_paths.crs != "EPSG:4326":
        all_paths = all_paths.to_crs("EPSG:4326")
    log.info("Loaded %d FlowPy paths (CRS: %s)", len(all_paths), all_paths.crs)

    results = []
    processed_ids = set()

    for src in CNN_SOURCES:
        log.info("Processing %s", src["name"])

        # Load seasonal probability cube if available
        prob_ds = None
        prob_da = None
        prob_times = None
        prob_bounds = None
        if src["seasonal_nc"] and src["seasonal_nc"].exists():
            prob_ds = xr.open_dataset(src["seasonal_nc"])
            prob_da = prob_ds["debris_probability"].rio.write_crs("EPSG:4326")
            prob_times = prob_da.time.values
            prob_bounds = prob_da.rio.bounds()
            log.info("  Seasonal NC: %d time steps, %s to %s",
                      len(prob_times), str(prob_times[0])[:10], str(prob_times[-1])[:10])

        # Load scene TIFs if available (for zones without seasonal NC)
        scene_tifs = {}
        if src["scene_tif_dir"] and src["scene_tif_dir"].exists():
            for tif in sorted(src["scene_tif_dir"].glob("scene_v2_debris_*.tif")):
                date_str = tif.stem.replace("scene_v2_debris_", "")
                scene_tifs[date_str] = tif
            log.info("  Scene TIFs: %d files", len(scene_tifs))

        # Load temporal onset if available
        onset_ds = None
        onset_bounds = None
        if src["temporal_onset_nc"] and src["temporal_onset_nc"].exists():
            onset_ds = xr.open_dataset(src["temporal_onset_nc"])
            for var in onset_ds.data_vars:
                onset_ds[var] = onset_ds[var].rio.write_crs("EPSG:4326")
            onset_bounds = onset_ds["onset_date"].rio.bounds()
            log.info("  Temporal onset loaded")

        if prob_da is None and not scene_tifs:
            log.info("  No CNN data available, skipping")
            continue

        n_matched = 0
        for _, path_row in all_paths.iterrows():
            obs_id = path_row["obs_id"]
            if obs_id in processed_ids:
                continue

            obs_date_str = path_row["date"]
            obs_date = pd.Timestamp(obs_date_str)
            path_geom = path_row.geometry

            # Look up original observation metadata
            obs_meta = obs_df[obs_df["id"] == obs_id]
            if obs_meta.empty:
                continue
            obs_meta = obs_meta.iloc[0]
            lng, lat = obs_meta["lng"], obs_meta["lat"]

            # ── Approach 1: Zonal stats at nearest time step ──
            prob_2d = None
            time_diff_days = np.nan

            if prob_da is not None and prob_bounds is not None:
                if (prob_bounds[0] <= lng <= prob_bounds[2]
                        and prob_bounds[1] <= lat <= prob_bounds[3]):
                    idx = find_nearest_time_idx(prob_times, obs_date)
                    time_diff_days = abs(
                        (prob_times[idx] - np.datetime64(obs_date)) / np.timedelta64(1, "D")
                    )
                    prob_2d = prob_da.isel(time=idx)

            elif scene_tifs:
                best_tif = None
                best_diff = float("inf")
                for date_str, tif_path in scene_tifs.items():
                    diff = abs((pd.Timestamp(date_str) - obs_date).days)
                    if diff < best_diff:
                        best_diff = diff
                        best_tif = tif_path
                        time_diff_days = diff
                if best_tif is not None:
                    tif_da = xr.open_dataarray(best_tif).rio.write_crs("EPSG:4326")
                    tif_bounds = tif_da.rio.bounds()
                    if (tif_bounds[0] <= lng <= tif_bounds[2]
                            and tif_bounds[1] <= lat <= tif_bounds[3]):
                        if "band" in tif_da.dims:
                            tif_da = tif_da.isel(band=0)
                        prob_2d = tif_da

            # Skip this obs if no CNN data covers it
            if prob_2d is None:
                continue

            # Skip if temporal gap exceeds threshold
            if args.max_time_gap is not None and time_diff_days > args.max_time_gap:
                continue

            processed_ids.add(obs_id)
            n_matched += 1

            record = {
                "obs_id": obs_id,
                "date": obs_date_str,
                "location_name": path_row.get("location_name", ""),
                "zone_name": path_row.get("zone_name", ""),
                "lat": lat,
                "lng": lng,
                "cnn_source": src["name"],
                "cnn_time_diff_days": float(time_diff_days),
            }

            zstats = zonal_stats_for_path(prob_2d, path_geom)
            record.update(zstats)

            # ── Approach 2: Onset date comparison ──
            if onset_ds is not None and onset_bounds is not None:
                if (onset_bounds[0] <= lng <= onset_bounds[2]
                        and onset_bounds[1] <= lat <= onset_bounds[3]):
                    ostats = onset_stats_for_path(onset_ds, path_geom, obs_date)
                    record.update(ostats)
                else:
                    record.update({
                        "n_candidates_in_path": 0,
                        "onset_date_diff_days_mean": np.nan,
                        "onset_date_diff_days_median": np.nan,
                        "onset_date_diff_days_min": np.nan,
                        "mean_confidence": np.nan,
                        "mean_peak_prob": np.nan,
                    })
            else:
                record.update({
                    "n_candidates_in_path": 0,
                    "onset_date_diff_days_mean": np.nan,
                    "onset_date_diff_days_median": np.nan,
                    "onset_date_diff_days_min": np.nan,
                    "mean_confidence": np.nan,
                    "mean_peak_prob": np.nan,
                })

            results.append(record)

        log.info("  Matched %d observations", n_matched)

        if prob_ds is not None:
            prob_ds.close()
        if onset_ds is not None:
            onset_ds.close()

    if not results:
        log.warning("No comparison results produced.")
        return

    result_df = pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════════════
    # DETECTION: spatial signal AND temporal match within ±6 days
    #
    # An observation counts as "detected" only if:
    #   1. max CNN prob inside path > max CNN prob in background annulus
    #   2. CNN onset date for at least one pixel inside the path is
    #      within ±ONSET_WINDOW days of the reported observation date
    #
    # For sources without onset data (Galena scene TIFs) we can only
    # evaluate criterion 1 — those are reported as "spatial_only".
    # ══════════════════════════════════════════════════════════════════════
    ONSET_WINDOW = 6  # days

    valid = result_df[result_df["max_prob_path"].notna()].copy()
    if len(valid) == 0:
        log.warning("No valid zonal stats.")
        return

    # Criterion 1: spatial — elevated probability in path vs background
    valid["spatial_detected"] = valid["max_prob_path"] > valid["max_prob_bg"]

    # Criterion 2: temporal — onset within ±ONSET_WINDOW days
    has_onset = valid["onset_date_diff_days_min"].notna()
    valid["onset_within_window"] = (
        has_onset & (valid["onset_date_diff_days_min"] <= ONSET_WINDOW)
    )

    # Combined detection: spatial + temporal (where onset data exists)
    # For sources with onset data: require both criteria
    # For sources without onset data: spatial-only (flagged separately)
    valid["has_onset_data"] = has_onset
    valid["detected"] = False
    with_onset = valid["has_onset_data"]
    valid.loc[with_onset, "detected"] = (
        valid.loc[with_onset, "spatial_detected"]
        & valid.loc[with_onset, "onset_within_window"]
    )
    valid.loc[~with_onset, "detected"] = valid.loc[~with_onset, "spatial_detected"]

    # Save
    valid.to_csv(OUT_CSV, index=False)
    log.info("Saved comparison summary to %s", OUT_CSV)

    # ── Report ────────────────────────────────────────────────────────────
    n_total = len(valid)
    n_with_onset = with_onset.sum()
    n_without_onset = n_total - n_with_onset

    print(f"\n{'='*70}")
    print(f"DETECTION RESULTS (buffer={args.path_buffer_m:.0f}m, "
          f"max_time_gap={args.max_time_gap}, onset_window=±{ONSET_WINDOW}d)")
    print(f"{'='*70}")
    print(f"Total observations: {n_total}")
    print(f"  With onset data:    {n_with_onset}")
    print(f"  Without onset data: {n_without_onset} (spatial-only evaluation)")
    print(f"Avg CNN time gap: {valid['cnn_time_diff_days'].mean():.1f} days")
    print(f"Median pixels per path: {valid['n_pixels_path'].median():.0f}")

    # ── Spatial detection (all observations) ──
    n_spatial = valid["spatial_detected"].sum()
    print(f"\nSpatial detection (max_prob_path > max_prob_bg):")
    print(f"  {n_spatial} / {n_total} ({100*n_spatial/n_total:.1f}%)")

    # ── Full detection (spatial + temporal, onset-data subset) ──
    onset_sub = valid[valid["has_onset_data"]]
    n_full = onset_sub["detected"].sum()
    n_spatial_only_onset = onset_sub["spatial_detected"].sum()
    print(f"\nFull detection (spatial + onset within ±{ONSET_WINDOW}d) "
          f"[{n_with_onset} obs with onset data]:")
    print(f"  Spatial detected: {n_spatial_only_onset} / {n_with_onset} "
          f"({100*n_spatial_only_onset/n_with_onset:.1f}%)")
    print(f"  + Onset match:    {n_full} / {n_with_onset} "
          f"({100*n_full/n_with_onset:.1f}%)")
    n_spatial_no_onset = n_spatial_only_onset - n_full
    print(f"  Spatial but wrong time: {n_spatial_no_onset} / {n_with_onset} "
          f"({100*n_spatial_no_onset/n_with_onset:.1f}%)")
    n_missed = n_with_onset - n_spatial_only_onset
    print(f"  Missed entirely:  {n_missed} / {n_with_onset} "
          f"({100*n_missed/n_with_onset:.1f}%)")

    # ── Spatial-only detection (no-onset subset, e.g. Galena) ──
    no_onset_sub = valid[~valid["has_onset_data"]]
    if len(no_onset_sub) > 0:
        n_sp = no_onset_sub["spatial_detected"].sum()
        print(f"\nSpatial-only detection [{n_without_onset} obs, no onset data]:")
        print(f"  {n_sp} / {n_without_onset} ({100*n_sp/n_without_onset:.1f}%)")

    # ── Detected event statistics ──
    det = valid[valid["detected"]]
    mis = valid[~valid["detected"]]
    print(f"\n--- Detected events (n={len(det)}) ---")
    if len(det) > 0:
        print(f"  Mean max prob in path:  {det['max_prob_path'].mean():.4f}")
        print(f"  Mean p90 prob in path:  {det['p90_prob_path'].mean():.4f}")
        print(f"  Mean prob inside path:  {det['mean_prob_path'].mean():.4f}")
        print(f"  Mean prob background:   {det['mean_prob_bg'].mean():.4f}")

    det_onset = det[det["has_onset_data"]]
    if len(det_onset) > 0:
        print(f"\n  Onset accuracy (n={len(det_onset)} with onset data):")
        print(f"    Mean onset diff:    {det_onset['onset_date_diff_days_mean'].mean():+.1f} days")
        print(f"    Median onset diff:  {det_onset['onset_date_diff_days_median'].median():+.1f} days")
        print(f"    Mean |closest px|:  {det_onset['onset_date_diff_days_min'].mean():.1f} days")
        print(f"    Mean confidence:    {det_onset['mean_confidence'].mean():.3f}")
        print(f"    Mean peak prob:     {det_onset['mean_peak_prob'].mean():.3f}")

    print(f"\n--- Missed events (n={len(mis)}) ---")
    if len(mis) > 0:
        print(f"  Mean max prob in path:  {mis['max_prob_path'].mean():.4f}")
        print(f"  Mean prob inside path:  {mis['mean_prob_path'].mean():.4f}")
        print(f"  Mean prob background:   {mis['mean_prob_bg'].mean():.4f}")

    # ── Breakdown by source ──
    print(f"\n--- By CNN source ---")
    for src_name, grp in valid.groupby("cnn_source"):
        nd = grp["detected"].sum()
        ns = grp["spatial_detected"].sum()
        ho = grp["has_onset_data"].sum()
        print(f"  {src_name} (n={len(grp)}, onset_data={ho}):")
        print(f"    Spatial: {ns}/{len(grp)} ({100*ns/len(grp):.0f}%)")
        if ho > 0:
            grp_o = grp[grp["has_onset_data"]]
            nd_o = grp_o["detected"].sum()
            print(f"    Full:    {nd_o}/{ho} ({100*nd_o/ho:.0f}%)")
        print(f"    Avg time gap: {grp['cnn_time_diff_days'].mean():.1f}d")


if __name__ == "__main__":
    main()
