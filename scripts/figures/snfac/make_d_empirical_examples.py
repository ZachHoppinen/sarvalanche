"""
Generate comparison figures: single-track diff vs d_empirical for labeled debris shapes.

For each debris shape, produces a 2-panel figure:
  Left:  single-track backscatter diff (closest track, one pair crossing avalanche date)
  Right: d_empirical (temporal+spatial weighted mean across all pairs and tracks)

Both panels show the debris outline overlaid.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw")
SHAPES_DIR = BASE / "debris_shapes"
NC_DIR = BASE / "netcdfs"
OUT_DIR = BASE / "figures" / "d_empirical_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map avalanche dates → netCDF file
DATE_TO_NC = {
    # Galena 2023-2024 season
    "2024-01-12": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",
    "2024-02-04": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",  # try Galena first
    "2024-02-29": NC_DIR / "Galena_Summit_&_Eastern_Mtns" / "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc",
    # Sawtooth 2024-2025 season
    "2024-11-15": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2024-12-29": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-02-04": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-02-19": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-03-15": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
    "2025-04-10": NC_DIR / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc",
}


# ── sarvalanche imports ────────────────────────────────────────────────────
sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))
from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.weights.temporal import get_temporal_weights
from sarvalanche.weights.combinations import weighted_mean
from sarvalanche.utils.validation import validate_weights_sum_to_one


def compute_d_empirical_for_track(da_track, avalanche_date, tau_days=6.0):
    """Compute temporal-weighted mean backscatter change for one track/pol."""
    diffs = backscatter_changes_crossing_date(da_track, avalanche_date)
    w = get_temporal_weights(diffs["t_start"], diffs["t_end"], tau_days=tau_days)
    validate_weights_sum_to_one(w, dim="pair")
    return weighted_mean(diffs, w, dim="pair")


def compute_single_diff_for_track(da_track, avalanche_date):
    """Compute the closest single before/after diff for one track."""
    times = pd.to_datetime(da_track.time.values)
    t0 = pd.to_datetime(avalanche_date)
    before = times[times <= t0]
    after = times[times > t0]
    if len(before) == 0 or len(after) == 0:
        return None
    t_before = before[-1]  # closest before
    t_after = after[0]     # closest after
    return da_track.sel(time=t_after) - da_track.sel(time=t_before)


def get_best_track(ds, shape_centroid):
    """Pick the track with the best resolution weight near the shape centroid."""
    # Use w_resolution to pick the best track
    w = ds["w_resolution"]
    # Find nearest pixel
    cx, cy = shape_centroid.x, shape_centroid.y
    w_at_point = w.sel(x=cx, y=cy, method="nearest")
    best_track = int(w_at_point.idxmax("static_track").values)
    return best_track


def process_date(date_str, nc_path, shapes_gdf):
    """Process all debris shapes for one avalanche date."""
    log.info(f"Processing date {date_str} with {len(shapes_gdf)} shapes")

    ds = xr.open_dataset(nc_path)
    avalanche_date = pd.Timestamp(date_str)

    # Get unique tracks
    tracks = np.unique(ds.track.values)
    static_tracks = ds.static_track.values

    for shape_idx, (_, shape_row) in enumerate(shapes_gdf.iterrows()):
        geom = shape_row.geometry
        centroid = geom.centroid

        # Get bounding box with buffer (0.02 degrees ~ 2km)
        minx, miny, maxx, maxy = geom.bounds
        buf = max(0.015, max(maxx - minx, maxy - miny) * 0.7)
        crop_box = box(minx - buf, miny - buf, maxx + buf, maxy + buf)
        cb = crop_box.bounds

        # Crop dataset to area around shape
        ds_crop = ds.sel(
            x=slice(cb[0], cb[2]),
            y=slice(cb[3], cb[1])  # y is typically decreasing
        )

        if ds_crop.sizes["x"] < 5 or ds_crop.sizes["y"] < 5:
            # Try other y order
            ds_crop = ds.sel(
                x=slice(cb[0], cb[2]),
                y=slice(cb[1], cb[3])
            )

        if ds_crop.sizes["x"] < 5 or ds_crop.sizes["y"] < 5:
            log.warning(f"  Shape {shape_idx}: crop too small, skipping")
            continue

        # Pick the best track for single diff
        best_track = get_best_track(ds, centroid)
        track_mask = ds_crop.track == best_track

        if track_mask.sum() < 2:
            # Fall back to most common track
            track_vals, counts = np.unique(ds_crop.track.values, return_counts=True)
            best_track = track_vals[np.argmax(counts)]
            track_mask = ds_crop.track == best_track

        da_vv_track = ds_crop["VV"].sel(time=track_mask)

        # Compute single diff
        single_diff = compute_single_diff_for_track(da_vv_track, avalanche_date)
        if single_diff is None:
            log.warning(f"  Shape {shape_idx}: no before/after for track {best_track}")
            continue

        # Compute d_empirical across all tracks
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
                    static_track=st,
                    x=centroid.x, y=centroid.y,
                    method="nearest"
                ).values)
                d_empirical_parts.append(d_emp)
                weights_parts.append(w_res)
            except Exception as e:
                log.debug(f"  Track {st} failed: {e}")
                continue

        if not d_empirical_parts:
            log.warning(f"  Shape {shape_idx}: no d_empirical computed")
            continue

        # Weighted mean across tracks
        w_arr = np.array(weights_parts)
        w_arr = w_arr / w_arr.sum()
        d_empirical = sum(d * w for d, w in zip(d_empirical_parts, w_arr))

        # ── Plot ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        vmin, vmax = 0.0, 4.0  # dB increase range for avalanche debris

        for ax, data, title in [
            (axes[0], single_diff, f"Single-track diff (track {best_track})"),
            (axes[1], d_empirical, "d_empirical (weighted multi-pair)"),
        ]:
            im = ax.pcolormesh(
                data.x, data.y, data.values,
                vmin=vmin, vmax=vmax,
                cmap="inferno",
                shading="auto",
            )
            # Overlay debris outline
            gdf_shape = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            gdf_shape.boundary.plot(ax=ax, color="cyan", linewidth=1.5)

            ax.set_title(title, fontsize=11)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

        fig.colorbar(im, ax=axes, label="Backscatter change (dB)", shrink=0.8)
        fig.suptitle(f"Avalanche date: {date_str} | Shape {shape_idx}", fontsize=13)

        out_path = OUT_DIR / f"{date_str}_shape{shape_idx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {out_path.name}")


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

    log.info(f"Done. Figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
