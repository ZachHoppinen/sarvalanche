"""Generate auto-labels from denoised SAR data on cold, clean dates.

Criteria for auto-labeling a pixel as debris:
  1. TV-denoised VV change > min_db (default 3 dB)
  2. Both pair endpoints are cold (HRRR t2m_max < max_temp, default -5°C)
  3. Not in water mask
  4. cell_counts > min_cell_counts (on a plausible avalanche path)
  5. Short pairs only (span <= max_span_days, default 15)

Outputs per-pair auto-label GeoPackages and a combined file.

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/generate_autolabels.py \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --hrrr local/issw/snfac/hrrr_temperature_sawtooth_2425.nc \
        --out-dir local/issw/debris_shapes/snfac/autolabels \
        --min-db 3.0 --max-temp -5.0 --min-cell-counts 1
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from rasterio.transform import from_bounds
from scipy.ndimage import label as ndlabel
from shapely.geometry import shape
from skimage.restoration import denoise_tv_chambolle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_hrrr_temp(hrrr_ds, date, hrrr_times):
    """Get scene-mean t2m_max for a date. Returns float or None."""
    diffs = np.abs(hrrr_times - date)
    ci = diffs.argmin()
    if diffs[ci].days > 2:
        return None
    return float(hrrr_ds["t2m_max"].isel(time=ci).values.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-db", type=float, default=3.0,
                        help="Minimum denoised VV increase (dB) to label as debris")
    parser.add_argument("--max-temp", type=float, default=-3.0,
                        help="Maximum HRRR t2m_max (°C) for both endpoints to be 'cold'")
    parser.add_argument("--min-cell-counts", type=float, default=100.0,
                        help="Minimum cell_counts (FlowPy) — must be on a runout path")
    parser.add_argument("--min-slope", type=float, default=0.05,
                        help="Minimum slope (radians) — exclude flat terrain")
    parser.add_argument("--max-span-days", type=int, default=60,
                        help="Maximum pair span in days")
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=20,
                        help="Minimum connected component size")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = xr.open_dataset(args.nc)
    H, W = ds.sizes["y"], ds.sizes["x"]
    x_arr, y_arr = ds.x.values, ds.y.values
    dx = abs(float(x_arr[1] - x_arr[0]))
    dy = abs(float(y_arr[1] - y_arr[0]))
    transform = from_bounds(
        float(x_arr.min()) - dx / 2, float(y_arr.min()) - dy / 2,
        float(x_arr.max()) + dx / 2, float(y_arr.max()) + dy / 2, W, H,
    )
    crs = ds.rio.crs if hasattr(ds, "rio") and ds.rio.crs else "EPSG:4326"

    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds.track.values

    # Static masks
    water = ds["water_mask"].values.astype(bool) if "water_mask" in ds else np.zeros((H, W), dtype=bool)
    cell_counts = ds["cell_counts"].values if "cell_counts" in ds else np.zeros((H, W))
    slope = ds["slope"].values if "slope" in ds else np.ones((H, W))

    terrain_mask = (~water) & (cell_counts >= args.min_cell_counts) & (slope >= args.min_slope)
    log.info("Terrain mask: %d valid pixels (%.1f%%)", terrain_mask.sum(), 100 * terrain_mask.mean())

    # HRRR
    hrrr_ds = xr.open_dataset(args.hrrr)
    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    # Group times by track
    track_entries = defaultdict(list)
    for ti, (t, tr) in enumerate(zip(times, tracks)):
        track_entries[int(tr)].append((t, ti))

    # Process each short pair
    all_pairs = []
    all_polys = []

    for tid, entries in sorted(track_entries.items()):
        entries.sort()
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                t_a, i_a = entries[a]
                t_b, i_b = entries[b]
                span = (t_b - t_a).days
                if span > args.max_span_days or span < 1:
                    continue

                # Check HRRR temperature at both endpoints
                temp_a = get_hrrr_temp(hrrr_ds, t_a, hrrr_times)
                temp_b = get_hrrr_temp(hrrr_ds, t_b, hrrr_times)
                if temp_a is None or temp_b is None:
                    continue
                if temp_a > args.max_temp or temp_b > args.max_temp:
                    continue

                # Load VV diff
                vv_a = ds["VV"].isel(time=int(i_a)).values
                vv_b = ds["VV"].isel(time=int(i_b)).values
                vv_diff = (vv_b - vv_a).astype(np.float32)
                valid = np.isfinite(vv_diff)

                # TV denoise
                filled = np.where(valid, vv_diff, 0.0).astype(np.float32)
                denoised = denoise_tv_chambolle(filled, weight=args.tv_weight).astype(np.float32)

                # Apply thresholds
                debris = (
                    (denoised >= args.min_db)
                    & valid
                    & terrain_mask
                )

                # Remove small components
                if debris.sum() == 0:
                    continue
                labeled, n_components = ndlabel(debris)
                for comp_id in range(1, n_components + 1):
                    if (labeled == comp_id).sum() < args.min_pixels:
                        debris[labeled == comp_id] = False

                n_debris = int(debris.sum())
                if n_debris == 0:
                    continue

                # Vectorize to polygons
                shapes_gen = rasterio.features.shapes(
                    debris.astype(np.uint8), mask=debris, transform=transform,
                )
                polys = []
                for geom, val in shapes_gen:
                    if val == 1:
                        polys.append(shape(geom))

                if not polys:
                    continue

                date_str = t_b.strftime("%Y-%m-%d")
                pair_label = f"trk{tid}_{t_a.strftime('%Y-%m-%d')}_{date_str}_{span}d"

                log.info("  %s: %d debris px, %d polygons, temp=[%.1f, %.1f]°C",
                         pair_label, n_debris, len(polys), temp_a, temp_b)

                for p in polys:
                    all_polys.append({
                        "geometry": p,
                        "track": tid,
                        "t_start": t_a.strftime("%Y-%m-%d"),
                        "t_end": date_str,
                        "span_days": span,
                        "temp_start": temp_a,
                        "temp_end": temp_b,
                        "pair_label": pair_label,
                    })
                all_pairs.append(pair_label)

    if not all_polys:
        log.info("No auto-labels found!")
        return

    # Save combined
    gdf = gpd.GeoDataFrame(all_polys, crs=crs)
    out_path = args.out_dir / "autolabels_combined.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    log.info("Saved %d polygons from %d pairs to %s", len(gdf), len(all_pairs), out_path)

    # Also save per-date files for training compatibility
    for date_str, group in gdf.groupby("t_end"):
        date_path = args.out_dir / f"autolabels_{date_str}.gpkg"
        group.to_file(date_path, driver="GPKG")
        log.info("  %s: %d polygons", date_str, len(group))

    # Summary
    log.info("\nSummary:")
    log.info("  Total polygons: %d", len(gdf))
    log.info("  Pairs with labels: %d", len(all_pairs))
    log.info("  Unique end dates: %d", gdf["t_end"].nunique())
    log.info("  Tracks: %s", sorted(gdf["track"].unique()))

    ds.close()
    hrrr_ds.close()


if __name__ == "__main__":
    main()
