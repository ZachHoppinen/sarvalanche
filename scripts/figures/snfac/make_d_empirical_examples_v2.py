"""
Generate comparison figures: single-pair diff vs d_empirical (temporal-weighted multi-pair).

For each debris shape, produces a 3-panel figure:
  Left:   single-pair backscatter diff (closest before/after pair, one track)
  Center: d_empirical for the same single track (temporal-weighted across all crossing pairs)
  Right:  d_empirical combined across all tracks (resolution-weighted)

This shows two levels of improvement:
  1. Single pair → multi-pair temporal weighting (reduces speckle noise)
  2. Single track → multi-track combination (further noise reduction + gap filling)
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw")
SHAPES_DIR = BASE / "debris_shapes"
NC_DIR = BASE / "netcdfs"
OUT_DIR = BASE / "figures" / "d_empirical_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_TO_NC = {
    "2024-01-12": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",
    "2024-02-04": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",
    "2024-02-29": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",
    "2024-11-15": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2024-12-29": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-02-04": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-02-19": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-03-15": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-04-10": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
}

MAX_SHAPES_PER_DATE = 40

sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))
from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.weights.temporal import get_temporal_weights
from sarvalanche.weights.combinations import weighted_mean
from sarvalanche.utils.validation import validate_weights_sum_to_one


def compute_d_empirical_for_track(da_track, avalanche_date, tau_days=6.0):
    """Temporal-weighted mean backscatter change for one track."""
    diffs = backscatter_changes_crossing_date(da_track, avalanche_date)
    w = get_temporal_weights(diffs["t_start"], diffs["t_end"], tau_days=tau_days)
    validate_weights_sum_to_one(w, dim="pair")
    return weighted_mean(diffs, w, dim="pair")


def compute_single_diff_for_track(da_track, avalanche_date):
    """Closest single before/after diff for one track."""
    times = pd.to_datetime(da_track.time.values)
    t0 = pd.to_datetime(avalanche_date)
    before = times[times <= t0]
    after = times[times > t0]
    if len(before) == 0 or len(after) == 0:
        return None, None, None
    t_before, t_after = before[-1], after[0]
    diff = da_track.sel(time=t_after) - da_track.sel(time=t_before)
    return diff, t_before, t_after


def get_best_track(ds, centroid):
    """Pick track with best resolution weight near centroid."""
    w = ds["w_resolution"]
    w_at_point = w.sel(x=centroid.x, y=centroid.y, method="nearest")
    return int(w_at_point.idxmax("static_track").values)


def nan_weighted_mean(arrays, weights):
    """Weighted mean ignoring NaN, returns NaN only where ALL inputs are NaN."""
    stacked = np.stack([a.values for a in arrays], axis=0)
    w = np.array(weights)[:, None, None]
    mask = ~np.isnan(stacked)
    w_masked = np.where(mask, w, 0.0)
    w_sum = w_masked.sum(axis=0)
    w_sum[w_sum == 0] = np.nan
    result = np.nansum(stacked * w_masked, axis=0) / w_sum
    # Return as DataArray with same coords as first input
    return xr.DataArray(result, dims=arrays[0].dims, coords=arrays[0].coords)


def process_date(date_str, nc_path, shapes_gdf):
    # Check existing
    existing = set(f.stem for f in OUT_DIR.glob(f"{date_str}_shape*.png"))

    if len(shapes_gdf) > MAX_SHAPES_PER_DATE:
        indices = np.linspace(0, len(shapes_gdf) - 1, MAX_SHAPES_PER_DATE, dtype=int)
        shapes_gdf = shapes_gdf.iloc[indices]

    todo = []
    for shape_idx, (_, row) in enumerate(shapes_gdf.iterrows()):
        key = f"{date_str}_shape{shape_idx:03d}"
        if key not in existing:
            todo.append((shape_idx, row))

    if not todo:
        log.info(f"  {date_str}: all done, skipping")
        return

    log.info(f"Processing {date_str}: {len(todo)} shapes to generate")
    sys.stdout.flush()

    ds = xr.open_dataset(nc_path)
    avalanche_date = pd.Timestamp(date_str)
    static_tracks = ds.static_track.values

    for shape_idx, shape_row in todo:
        geom = shape_row.geometry
        centroid = geom.centroid

        minx, miny, maxx, maxy = geom.bounds
        buf = max(0.015, max(maxx - minx, maxy - miny) * 0.7)
        cb = (minx - buf, miny - buf, maxx + buf, maxy + buf)

        ds_crop = ds.sel(x=slice(cb[0], cb[2]), y=slice(cb[3], cb[1]))
        if ds_crop.sizes["x"] < 5 or ds_crop.sizes["y"] < 5:
            ds_crop = ds.sel(x=slice(cb[0], cb[2]), y=slice(cb[1], cb[3]))
        if ds_crop.sizes["x"] < 5 or ds_crop.sizes["y"] < 5:
            log.warning(f"  Shape {shape_idx}: crop too small")
            continue

        # Best track for single-pair panel
        best_track = get_best_track(ds, centroid)
        track_mask = ds_crop.track == best_track
        if track_mask.sum() < 2:
            track_vals, counts = np.unique(ds_crop.track.values, return_counts=True)
            best_track = track_vals[np.argmax(counts)]
            track_mask = ds_crop.track == best_track

        da_vv_track = ds_crop["VV"].sel(time=track_mask)

        # Panel 1: single diff
        single_diff, t_before, t_after = compute_single_diff_for_track(da_vv_track, avalanche_date)
        if single_diff is None:
            continue

        # Panel 2: d_empirical for same track (multi-pair temporal weighting)
        try:
            d_emp_single_track = compute_d_empirical_for_track(da_vv_track, avalanche_date)
        except Exception:
            continue

        # Panel 3: d_empirical combined across all tracks
        d_empirical_parts = []
        weights_parts = []
        for st in static_tracks:
            tmask = ds_crop.track == st
            if tmask.sum() < 2:
                continue
            da_vv_st = ds_crop["VV"].sel(time=tmask)
            times = pd.to_datetime(da_vv_st.time.values)
            t0 = pd.to_datetime(avalanche_date)
            if not (any(times <= t0) and any(times > t0)):
                continue
            try:
                d_emp = compute_d_empirical_for_track(da_vv_st, avalanche_date)
                w_res = float(ds_crop["w_resolution"].sel(
                    static_track=st, x=centroid.x, y=centroid.y, method="nearest"
                ).values)
                d_empirical_parts.append(d_emp)
                weights_parts.append(max(w_res, 1e-6))
            except Exception:
                continue

        if not d_empirical_parts:
            continue

        # NaN-safe weighted mean across tracks
        w_arr = np.array(weights_parts)
        w_arr = w_arr / w_arr.sum()
        d_empirical_combined = nan_weighted_mean(d_empirical_parts, w_arr)

        # ── Plot ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        vmin, vmax = 0.0, 4.0
        cmap = "inferno"

        panels = [
            (axes[0], single_diff,
             f"Single pair (track {best_track})\n{str(t_before.date())} → {str(t_after.date())}"),
            (axes[1], d_emp_single_track,
             f"d_empirical single track ({best_track})\n(temporal-weighted, all pairs)"),
            (axes[2], d_empirical_combined,
             f"d_empirical all tracks\n(temporal + resolution weighted)"),
        ]

        for ax, data, title in panels:
            im = ax.pcolormesh(
                data.x, data.y, data.values,
                vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
            )
            gdf_shape = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            gdf_shape.boundary.plot(ax=ax, color="cyan", linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

        fig.colorbar(im, ax=axes, label="Backscatter change (dB)", shrink=0.8, pad=0.02)
        fig.suptitle(f"Avalanche date: {date_str} | Shape {shape_idx}", fontsize=13)

        out_path = OUT_DIR / f"{date_str}_shape{shape_idx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {out_path.name}")
        sys.stdout.flush()

    ds.close()


def main():
    shape_files = sorted(SHAPES_DIR.glob("avalanche_labels_*.gpkg"))
    for sf in shape_files:
        date_str = sf.stem.replace("avalanche_labels_", "")
        nc_path = DATE_TO_NC.get(date_str)
        if nc_path is None or not nc_path.exists():
            log.warning(f"No netCDF for {date_str}, skipping")
            continue
        shapes_gdf = gpd.read_file(sf)
        if shapes_gdf.crs is None:
            shapes_gdf = shapes_gdf.set_crs("EPSG:4326")
        elif shapes_gdf.crs.to_epsg() != 4326:
            shapes_gdf = shapes_gdf.to_crs("EPSG:4326")
        process_date(date_str, nc_path, shapes_gdf)

    log.info(f"Done. All figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
