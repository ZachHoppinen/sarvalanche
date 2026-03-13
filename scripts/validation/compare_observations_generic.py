#!/usr/bin/env python3
"""Compare CNN debris probabilities against FlowPy observation paths — generic.

Works with any avalanche center. Auto-discovers CNN inference outputs from
zone directories under --nc-dir.

Usage:
    conda run -n sarvalanche python scripts/validation/compare_observations_generic.py \
        --obs-csv local/cnfaic/cnfaic_obs_all.csv \
        --paths-gpkg local/cnfaic/observations/all_flowpy_paths.gpkg \
        --nc-dir local/cnfaic/netcdfs \
        --out-csv local/cnfaic/observations/comparison_summary.csv
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
from rasterio.transform import from_bounds

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PATH_BUFFER_M = 500
BACKGROUND_BUFFER_DEG = 0.005
ONSET_WINDOW = 6  # days


def discover_cnn_sources(nc_dir: Path) -> list[dict]:
    sources = []
    for zone_dir in sorted(nc_dir.iterdir()):
        if not zone_dir.is_dir():
            continue
        zone_name = zone_dir.name

        inf_dirs = sorted(zone_dir.glob("v2_season_inference*"))
        v2_inf = zone_dir / "v2_inference"
        if v2_inf.is_dir() and v2_inf not in inf_dirs:
            inf_dirs.append(v2_inf)

        for inf_dir in inf_dirs:
            seasonal_nc = inf_dir / "season_v2_debris_probabilities.nc"
            temporal_nc = inf_dir / "temporal_onset.nc"
            scene_tifs = list(inf_dir.glob("scene_v2_debris_*.tif"))

            if seasonal_nc.exists() or scene_tifs:
                name = zone_name
                if inf_dir.name != "v2_season_inference" and inf_dir.name != "v2_inference":
                    season_suffix = inf_dir.name.replace("v2_season_inference_", "")
                    name = f"{zone_name}_{season_suffix}"

                sources.append({
                    "name": name,
                    "seasonal_nc": seasonal_nc if seasonal_nc.exists() else None,
                    "temporal_onset_nc": temporal_nc if temporal_nc.exists() else None,
                    "scene_tif_dir": inf_dir if scene_tifs else None,
                })
    return sources


def parse_location_point(loc_str: str) -> tuple[float, float]:
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def load_observations_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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


def rasterize_path_to_grid(path_geom, da: xr.DataArray) -> np.ndarray:
    ny, nx = da.sizes["y"], da.sizes["x"]
    bounds = da.rio.bounds()
    transform = from_bounds(*bounds, nx, ny)
    mask = rasterio.features.geometry_mask(
        [path_geom], out_shape=(ny, nx), transform=transform, invert=True,
    )
    return mask


def zonal_stats_for_path(prob_2d, path_geom) -> dict:
    path_mask = rasterize_path_to_grid(path_geom, prob_2d)
    prob_vals = prob_2d.values
    inside = prob_vals[path_mask]
    inside = inside[np.isfinite(inside)]

    if len(inside) == 0:
        return {
            "n_pixels_path": 0, "max_prob_path": np.nan,
            "mean_prob_path": np.nan, "p90_prob_path": np.nan,
            "p95_prob_path": np.nan, "n_pixels_above_05": 0,
            "n_pixels_bg": 0, "mean_prob_bg": np.nan,
            "max_prob_bg": np.nan, "p95_prob_bg": np.nan,
        }

    bg_geom = path_geom.buffer(BACKGROUND_BUFFER_DEG).difference(path_geom)
    bg_mask = rasterize_path_to_grid(bg_geom, prob_2d)
    outside = prob_vals[bg_mask]
    outside = outside[np.isfinite(outside)]

    return {
        "n_pixels_path": len(inside),
        "max_prob_path": float(np.max(inside)),
        "mean_prob_path": float(np.mean(inside)),
        "p90_prob_path": float(np.percentile(inside, 90)),
        "p95_prob_path": float(np.percentile(inside, 95)),
        "n_pixels_above_05": int((inside > 0.5).sum()),
        "n_pixels_bg": len(outside),
        "mean_prob_bg": float(np.mean(outside)) if len(outside) > 0 else np.nan,
        "max_prob_bg": float(np.max(outside)) if len(outside) > 0 else np.nan,
        "p95_prob_bg": float(np.percentile(outside, 95)) if len(outside) > 0 else np.nan,
    }


def onset_stats_for_path(onset_ds, path_geom, obs_date) -> dict:
    onset_da = onset_ds["onset_date"]
    candidate_da = onset_ds["candidate_mask"]
    confidence_da = onset_ds["confidence"]
    peak_prob_da = onset_ds["peak_prob"]

    path_mask = rasterize_path_to_grid(path_geom, onset_da)
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

    obs_dt64 = np.datetime64(obs_date)

    has_multi = "all_onset_dates" in onset_ds
    if has_multi:
        all_dates = onset_ds["all_onset_dates"].values
        K = all_dates.shape[0]
        all_diffs = []
        for k in range(K):
            k_dates = all_dates[k][combined]
            valid_k = ~np.isnat(k_dates)
            if valid_k.sum() > 0:
                k_diffs = (k_dates[valid_k] - obs_dt64) / np.timedelta64(1, "D")
                all_diffs.append(k_diffs)
        all_diffs = np.concatenate(all_diffs) if all_diffs else np.array([])
    else:
        onset_dates = onset_da.values[combined]
        valid = ~np.isnat(onset_dates)
        onset_dates = onset_dates[valid]
        all_diffs = (onset_dates - obs_dt64) / np.timedelta64(1, "D") if len(onset_dates) > 0 else np.array([])

    if len(all_diffs) == 0:
        return {
            "n_candidates_in_path": n_candidates,
            "onset_date_diff_days_mean": np.nan,
            "onset_date_diff_days_median": np.nan,
            "onset_date_diff_days_min": np.nan,
            "mean_confidence": np.nan,
            "mean_peak_prob": np.nan,
        }

    conf_vals = confidence_da.values[combined]
    peak_vals = peak_prob_da.values[combined]

    return {
        "n_candidates_in_path": n_candidates,
        "onset_date_diff_days_mean": float(np.nanmean(all_diffs)),
        "onset_date_diff_days_median": float(np.nanmedian(all_diffs)),
        "onset_date_diff_days_min": float(np.min(np.abs(all_diffs))),
        "mean_confidence": float(np.nanmean(conf_vals)),
        "mean_peak_prob": float(np.nanmean(peak_vals)),
    }


def find_nearest_time_idx(times, target):
    diffs = np.abs(times - np.datetime64(target))
    return int(np.argmin(diffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-csv", type=Path, required=True, help="Observations CSV")
    parser.add_argument("--paths-gpkg", type=Path, required=True, help="FlowPy paths GeoPackage")
    parser.add_argument("--nc-dir", type=Path, required=True, help="NetCDF directory with zone subdirs")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output comparison CSV")
    parser.add_argument("--max-time-gap", type=float, default=None)
    parser.add_argument("--path-buffer-m", type=float, default=PATH_BUFFER_M)
    args = parser.parse_args()

    obs_df = load_observations_df(args.obs_csv)
    all_paths = gpd.read_file(args.paths_gpkg)

    if args.path_buffer_m > 0:
        log.info("Buffering FlowPy paths by %.0f m (in %s)", args.path_buffer_m, all_paths.crs)
        all_paths["geometry"] = all_paths.geometry.buffer(args.path_buffer_m)
    if all_paths.crs and all_paths.crs != "EPSG:4326":
        all_paths = all_paths.to_crs("EPSG:4326")
    log.info("Loaded %d FlowPy paths (CRS: %s)", len(all_paths), all_paths.crs)

    cnn_sources = discover_cnn_sources(args.nc_dir)
    log.info("Discovered %d CNN sources", len(cnn_sources))

    results = []
    processed_ids = set()

    for src in cnn_sources:
        log.info("Processing %s", src["name"])

        prob_ds = prob_da = prob_times = prob_bounds = None
        if src["seasonal_nc"] and src["seasonal_nc"].exists():
            prob_ds = xr.open_dataset(src["seasonal_nc"])
            prob_da = prob_ds["debris_probability"].rio.write_crs("EPSG:4326")
            prob_times = prob_da.time.values
            prob_bounds = prob_da.rio.bounds()

        scene_tifs = {}
        if src["scene_tif_dir"] and src["scene_tif_dir"].exists():
            for tif in sorted(src["scene_tif_dir"].glob("scene_v2_debris_*.tif")):
                date_str = tif.stem.replace("scene_v2_debris_", "")
                scene_tifs[date_str] = tif

        onset_ds = onset_bounds = None
        if src["temporal_onset_nc"] and src["temporal_onset_nc"].exists():
            onset_ds = xr.open_dataset(src["temporal_onset_nc"])
            for var in onset_ds.data_vars:
                onset_ds[var] = onset_ds[var].rio.write_crs("EPSG:4326")
            onset_bounds = onset_ds["onset_date"].rio.bounds()

        if prob_da is None and not scene_tifs:
            continue

        n_matched = 0
        for _, path_row in all_paths.iterrows():
            obs_id = path_row["obs_id"]
            if obs_id in processed_ids:
                continue

            obs_date_str = path_row["date"]
            obs_date = pd.Timestamp(obs_date_str)
            path_geom = path_row.geometry

            obs_meta = obs_df[obs_df["id"] == obs_id]
            if obs_meta.empty:
                continue
            obs_meta = obs_meta.iloc[0]
            lng, lat = obs_meta["lng"], obs_meta["lat"]

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

            if prob_2d is None:
                continue
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

            record.update(zonal_stats_for_path(prob_2d, path_geom))

            if onset_ds is not None and onset_bounds is not None:
                if (onset_bounds[0] <= lng <= onset_bounds[2]
                        and onset_bounds[1] <= lat <= onset_bounds[3]):
                    record.update(onset_stats_for_path(onset_ds, path_geom, obs_date))
                else:
                    record.update({k: np.nan for k in [
                        "onset_date_diff_days_mean", "onset_date_diff_days_median",
                        "onset_date_diff_days_min", "mean_confidence", "mean_peak_prob",
                    ]})
                    record["n_candidates_in_path"] = 0
            else:
                record.update({k: np.nan for k in [
                    "onset_date_diff_days_mean", "onset_date_diff_days_median",
                    "onset_date_diff_days_min", "mean_confidence", "mean_peak_prob",
                ]})
                record["n_candidates_in_path"] = 0

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

    # Detection logic
    valid = result_df[result_df["max_prob_path"].notna()].copy()
    if len(valid) == 0:
        log.warning("No valid zonal stats.")
        return

    valid["spatial_detected"] = (
        (valid["p95_prob_path"] > valid["p95_prob_bg"])
        | (valid["n_pixels_above_05"] >= 10)
    )

    has_onset = valid["onset_date_diff_days_min"].notna()
    valid["onset_within_window"] = (
        has_onset & (valid["onset_date_diff_days_min"] <= ONSET_WINDOW)
    )

    valid["has_onset_data"] = has_onset
    valid["detected"] = False
    with_onset = valid["has_onset_data"]
    valid.loc[with_onset, "detected"] = (
        valid.loc[with_onset, "spatial_detected"]
        & valid.loc[with_onset, "onset_within_window"]
    )
    valid.loc[~with_onset, "detected"] = valid.loc[~with_onset, "spatial_detected"]

    # Merge d_size from observations
    obs_dsize = load_observations_df(args.obs_csv)[["id", "d_size"]].rename(columns={"id": "obs_id"})
    valid = valid.merge(obs_dsize, on="obs_id", how="left")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(args.out_csv, index=False)
    log.info("Saved comparison summary to %s", args.out_csv)

    # ── Report ────────────────────────────────────────────────────────────
    n_total = len(valid)
    n_with_onset = int(with_onset.sum())
    n_without_onset = n_total - n_with_onset

    print(f"\n{'='*70}")
    print(f"DETECTION RESULTS (buffer={args.path_buffer_m:.0f}m, "
          f"max_time_gap={args.max_time_gap}, onset_window=±{ONSET_WINDOW}d)")
    print(f"{'='*70}")
    print(f"Total observations: {n_total}")

    # ── Overall detection rates ──
    n_spatial = int(valid["spatial_detected"].sum())
    n_detected = int(valid["detected"].sum())
    print(f"\nSpatial detection: {n_spatial} / {n_total} ({100*n_spatial/n_total:.1f}%)")
    print(f"Full detection:    {n_detected} / {n_total} ({100*n_detected/n_total:.1f}%)")

    # ── By D-size ──
    print(f"\n--- Detection rate by D-size ---")
    for dsize in sorted(valid["d_size"].dropna().unique()):
        sub = valid[valid["d_size"] == dsize]
        n_d = len(sub)
        n_sp = int(sub["spatial_detected"].sum())
        n_det = int(sub["detected"].sum())
        print(f"  D{dsize}: {n_det}/{n_d} ({100*n_det/n_d:.0f}%) detected, "
              f"{n_sp}/{n_d} ({100*n_sp/n_d:.0f}%) spatial")

    # ── By D-size groups ──
    print(f"\n--- Detection rate by D-size groups ---")
    for label, mask in [
        ("D1-D1.5", valid["d_size"] <= 1.5),
        ("D2-D2.5", (valid["d_size"] >= 2) & (valid["d_size"] <= 2.5)),
        ("D3+", valid["d_size"] >= 3),
    ]:
        sub = valid[mask]
        if len(sub) == 0:
            continue
        n_d = len(sub)
        n_sp = int(sub["spatial_detected"].sum())
        n_det = int(sub["detected"].sum())
        print(f"  {label}: {n_det}/{n_d} ({100*n_det/n_d:.0f}%) detected, "
              f"{n_sp}/{n_d} ({100*n_sp/n_d:.0f}%) spatial")

    # ── By zone ──
    print(f"\n--- Detection rate by zone ---")
    for zone, grp in valid.groupby("zone_name"):
        n_z = len(grp)
        n_det = int(grp["detected"].sum())
        n_sp = int(grp["spatial_detected"].sum())
        print(f"  {zone} (n={n_z}): {n_det}/{n_z} ({100*n_det/n_z:.0f}%) detected, "
              f"{n_sp}/{n_z} ({100*n_sp/n_z:.0f}%) spatial")

    # ── By month ──
    valid["month"] = pd.to_datetime(valid["date"]).dt.month
    print(f"\n--- Detection rate by month ---")
    for month in sorted(valid["month"].unique()):
        sub = valid[valid["month"] == month]
        n_m = len(sub)
        n_det = int(sub["detected"].sum())
        n_sp = int(sub["spatial_detected"].sum())
        month_name = pd.Timestamp(f"2025-{month:02d}-01").strftime("%b")
        print(f"  {month_name} (n={n_m}): {n_det}/{n_m} ({100*n_det/n_m:.0f}%) detected, "
              f"{n_sp}/{n_m} ({100*n_sp/n_m:.0f}%) spatial")

    # ── By CNN source ──
    print(f"\n--- By CNN source ---")
    for src_name, grp in valid.groupby("cnn_source"):
        nd = int(grp["detected"].sum())
        ns = int(grp["spatial_detected"].sum())
        print(f"  {src_name} (n={len(grp)}): {nd}/{len(grp)} ({100*nd/len(grp):.0f}%) detected, "
              f"{ns}/{len(grp)} ({100*ns/len(grp):.0f}%) spatial")

    # ── Detected vs missed stats ──
    det = valid[valid["detected"]]
    mis = valid[~valid["detected"]]
    print(f"\n--- Detected events (n={len(det)}) ---")
    if len(det) > 0:
        print(f"  Mean max prob in path:  {det['max_prob_path'].mean():.4f}")
        print(f"  Mean prob inside path:  {det['mean_prob_path'].mean():.4f}")
        print(f"  Mean prob background:   {det['mean_prob_bg'].mean():.4f}")

    print(f"\n--- Missed events (n={len(mis)}) ---")
    if len(mis) > 0:
        print(f"  Mean max prob in path:  {mis['max_prob_path'].mean():.4f}")
        print(f"  Mean prob inside path:  {mis['mean_prob_path'].mean():.4f}")
        print(f"  Mean prob background:   {mis['mean_prob_bg'].mean():.4f}")


if __name__ == "__main__":
    main()
