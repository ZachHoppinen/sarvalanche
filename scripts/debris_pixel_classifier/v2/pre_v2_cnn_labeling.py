"""Pre-labeling setup: build season dataset + export GeoTIFF windows for QGIS labeling.

For each date in --dates:
  1. Compute empirical layers
  2. Select 2 windows (~30 km each): 1 high-probability, 1 low-probability
  3. Export d_empirical GeoTIFFs to debris_shapes/geotiffs/<date>/
  4. Create seed avalanche_labels_<date>.gpkg for polygon drawing in QGIS
  5. Update patch_footprints.gpkg

Prerequisite: season_dataset.nc must exist (run build_season_dataset.py first),
or pass --build-nc to build it automatically.

Usage:
    # With existing season_dataset.nc:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/pre_v2_cnn_labeling.py \
        --nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
        --dates 2024-12-29 2025-02-04 2025-02-19 \
        --out-dir local/issw/debris_shapes

    # Build season_dataset.nc first, then export:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/pre_v2_cnn_labeling.py \
        --zone "Sawtooth & Western Smoky Mtns" \
        --season 2024-2025 \
        --nc-dir local/issw/dual_tau_output/ \
        --dates 2024-12-29 2025-02-04 2025-02-19 \
        --out-dir local/issw/debris_shapes
"""

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

TAU = 6
N_PATCHES = 2
PATCH_KM = 30


# ---------------------------------------------------------------------------
# Empirical computation (same as other scripts)
# ---------------------------------------------------------------------------

def compute_empirical_for_date(ds, reference_date, tau_days=TAU):
    """Compute per-track/pol empirical layers for a given date."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)

    stale_patterns = [
        re.compile(r"^p_\d+_V[VH]_empirical$"),
        re.compile(r"^d_\d+_V[VH]_empirical$"),
    ]
    stale_exact = {"p_empirical", "d_empirical", "w_temporal"}
    to_drop = [v for v in ds.data_vars if v in stale_exact]
    for pat in stale_patterns:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        ds = ds.drop_vars(to_drop)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_empirical, d_empirical = calculate_empirical_backscatter_probability(
        ds, ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical
    return ds


# ---------------------------------------------------------------------------
# Window selection (adapted from patch_labeler.py)
# ---------------------------------------------------------------------------

def _get_resolution_m(ds):
    """Get approximate pixel resolution in meters."""
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        mid_lat = float(ds.y.values.mean())
        dx_m = dx * 111320 * np.cos(np.radians(mid_lat))
        dy_m = dy * 110540
        return (dx_m + dy_m) / 2
    return (dx + dy) / 2


def build_window_grid(ds, window_px):
    """Build non-overlapping grid of (y0, x0) window origins."""
    H, W = ds.sizes["y"], ds.sizes["x"]
    coords = []
    for y0 in range(0, H - window_px + 1, window_px):
        for x0 in range(0, W - window_px + 1, window_px):
            coords.append((y0, x0))
    return coords


def score_windows(ds, coords, window_px):
    """Compute mean p_empirical per window."""
    p = ds["p_empirical"].values
    scored = []
    for y0, x0 in coords:
        patch = p[y0:y0 + window_px, x0:x0 + window_px]
        mean_p = float(np.nanmean(patch))
        frac_valid = float(np.isfinite(patch).mean())
        if frac_valid < 0.5:
            continue
        scored.append((y0, x0, mean_p))
    return scored


def select_windows(scored, n_patches):
    """Select n_patches: half highest mean_p, half lowest."""
    scored_sorted = sorted(scored, key=lambda x: x[2])
    n_half = n_patches // 2

    low = scored_sorted[:n_half]
    high = scored_sorted[-n_half:]

    selected = []
    for h, l in zip(high, low):
        selected.append((*h, "high"))
        selected.append((*l, "low"))
    if n_patches % 2 == 1 and len(scored_sorted) > n_patches:
        mid_idx = len(scored_sorted) // 2
        selected.append((*scored_sorted[mid_idx], "mid"))

    return selected


# ---------------------------------------------------------------------------
# GeoTIFF export (adapted from patch_labeler.py)
# ---------------------------------------------------------------------------

def save_geotiff_windows(ds, selected, window_px, out_dir, date_str, tau):
    """Save d_empirical windows as GeoTIFFs + footprints + seed labels gpkg."""
    tau_str = f"tau{tau:g}"
    geotiff_dir = out_dir / "geotiffs" / date_str
    geotiff_dir.mkdir(parents=True, exist_ok=True)

    crs = ds.rio.crs
    footprints = []
    has_water = "water_mask" in ds.data_vars

    for i, (y0, x0, mean_p, bucket) in enumerate(selected):
        window = ds["d_empirical"].isel(
            y=slice(y0, y0 + window_px),
            x=slice(x0, x0 + window_px),
        ).copy()
        if has_water:
            water = ds["water_mask"].isel(
                y=slice(y0, y0 + window_px),
                x=slice(x0, x0 + window_px),
            )
            window = window.where(water == 0, 0)
        fname = geotiff_dir / f"d_empirical_{date_str}_{tau_str}_{bucket}_{y0:04d}_{x0:04d}.tif"
        window.rio.to_raster(str(fname))
        log.info(
            "  [%d/%d] Saved %s  (mean_p=%.3f, %s)",
            i + 1, len(selected), fname.name, mean_p, bucket,
        )

        x_vals = ds.x.values[x0:x0 + window_px]
        y_vals = ds.y.values[y0:y0 + window_px]
        footprints.append({
            "geometry": box(
                float(x_vals.min()), float(y_vals.min()),
                float(x_vals.max()), float(y_vals.max()),
            ),
            "patch_id": f"{bucket}_{y0:04d}_{x0:04d}",
            "date": date_str,
            "mean_p": mean_p,
            "bucket": bucket,
            "geotiff": fname.name,
            "reviewed": False,
        })

    # Seed labeling GeoPackage
    labels_path = out_dir / f"avalanche_labels_{date_str}.gpkg"
    if not labels_path.exists():
        y0, x0 = selected[0][0], selected[0][1]
        cx = float(ds.x.values[x0 + window_px // 2])
        cy = float(ds.y.values[y0 + window_px // 2])
        res = abs(float(ds.x.values[1] - ds.x.values[0]))
        seed = gpd.GeoDataFrame(
            geometry=[box(cx, cy, cx + res, cy + res)],
            crs=crs,
        )
        seed.to_file(labels_path, driver="GPKG")
        log.info("  Created labeling GeoPackage: %s", labels_path)
    else:
        log.info("  Labeling GeoPackage already exists: %s", labels_path)

    # Append to patch_footprints.gpkg
    new_gdf = gpd.GeoDataFrame(footprints, crs=crs)
    footprints_path = out_dir / "patch_footprints.gpkg"
    if footprints_path.exists():
        existing = gpd.read_file(footprints_path)
        existing = existing[existing["date"] != date_str]
        combined = pd.concat([existing, new_gdf], ignore_index=True)
    else:
        combined = new_gdf
    combined.to_file(footprints_path, driver="GPKG")
    log.info("  Footprints saved (%d total rows)", len(combined))

    return len(selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-labeling: build season dataset + export GeoTIFF windows for QGIS",
    )

    # Dataset source: either provide --nc directly, or --zone + --season to build it
    nc_group = parser.add_mutually_exclusive_group(required=True)
    nc_group.add_argument(
        "--nc", type=Path,
        help="Path to existing season_dataset.nc",
    )
    nc_group.add_argument(
        "--zone", type=str,
        help='SNFAC zone name (triggers build_season_dataset)',
    )

    parser.add_argument(
        "--season", type=str, default=None,
        help='Winter season "YYYY-YYYY" (required with --zone)',
    )
    parser.add_argument(
        "--nc-dir", type=Path, default=Path("./local/issw/dual_tau_output/"),
        help="Output directory for season_dataset.nc (used with --zone)",
    )
    parser.add_argument(
        "--existing-runs-dir", type=Path, default=None,
        help="Existing sarvalanche_runs/ dir to reuse FlowPy files (used with --zone)",
    )
    parser.add_argument(
        "--dates", nargs="+", required=True,
        help="Reference dates to export (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("local/issw/debris_shapes"),
        help="Output directory for GeoTIFFs and label gpkgs",
    )
    parser.add_argument(
        "--n-patches", type=int, default=N_PATCHES,
        help=f"Number of windows per date (default: {N_PATCHES})",
    )
    parser.add_argument(
        "--patch-km", type=float, default=PATCH_KM,
        help=f"Window size in km (default: {PATCH_KM})",
    )
    parser.add_argument(
        "--tau", type=float, default=TAU,
        help=f"Temporal decay tau in days (default: {TAU})",
    )
    args = parser.parse_args()

    # ── Resolve season_dataset.nc ────────────────────────────────────────
    if args.zone:
        if not args.season:
            parser.error("--season is required when using --zone")

        safe_name = args.zone.replace(" ", "_").replace("/", "-")
        zone_cache = args.nc_dir / safe_name
        season_nc = zone_cache / "season_dataset.nc"

        if not season_nc.exists():
            log.info("Building season dataset for %s %s...", args.zone, args.season)
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "scripts/issw_analysis/build_season_dataset.py",
                "--zone", args.zone,
                "--season", args.season,
                "--out-dir", str(args.nc_dir),
            ]
            if args.existing_runs_dir:
                cmd += ["--existing-runs-dir", str(args.existing_runs_dir)]

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                log.error("build_season_dataset.py failed (exit %d)", result.returncode)
                return
        else:
            log.info("Season dataset already exists: %s", season_nc)

        nc_path = season_nc
    else:
        nc_path = args.nc

    if not nc_path.exists():
        log.error("Season dataset not found: %s", nc_path)
        return

    # ── Load dataset ─────────────────────────────────────────────────────
    log.info("Loading %s", nc_path)
    ds = load_netcdf_to_dataset(nc_path)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  %d time steps, %d×%d spatial", len(ds.time), ds.sizes["y"], ds.sizes["x"])

    if "w_resolution" not in ds.data_vars:
        log.error("w_resolution not found — run build_season_dataset.py first")
        return

    # ── Compute window size ──────────────────────────────────────────────
    V2_PATCH_SIZE = 128
    res_m = _get_resolution_m(ds)
    window_px = int(round(args.patch_km * 1000 / res_m))
    window_px = max(V2_PATCH_SIZE, (window_px // V2_PATCH_SIZE) * V2_PATCH_SIZE)
    log.info(
        "Resolution: %.1f m/px → window = %d px (%.1f km)",
        res_m, window_px, window_px * res_m / 1000,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each date ────────────────────────────────────────────────
    total_windows = 0
    for date_str in args.dates:
        log.info("=== Date: %s ===", date_str)

        # Compute empirical for this date
        ref_date = pd.Timestamp(date_str)
        ds = compute_empirical_for_date(ds, ref_date, args.tau)

        # Build window grid and score
        coords = build_window_grid(ds, window_px)
        scored = score_windows(ds, coords, window_px)
        log.info("  %d valid windows out of %d grid cells", len(scored), len(coords))

        if not scored:
            log.warning("  No valid windows for %s, skipping", date_str)
            continue

        n = min(args.n_patches, len(scored))
        selected = select_windows(scored, n)
        log.info(
            "  Selected %d windows: %d high, %d low",
            len(selected),
            sum(1 for *_, b in selected if b == "high"),
            sum(1 for *_, b in selected if b == "low"),
        )

        n_saved = save_geotiff_windows(ds, selected, window_px, args.out_dir, date_str, args.tau)
        total_windows += n_saved

    log.info(
        "Done. Exported %d windows across %d dates to %s",
        total_windows, len(args.dates), args.out_dir,
    )
    log.info("")
    log.info("Next steps:")
    log.info("  1. Open GeoTIFFs in QGIS: %s/geotiffs/", args.out_dir)
    log.info("  2. Draw debris polygons in: %s/avalanche_labels_<date>.gpkg", args.out_dir)
    log.info("  3. Run post_v2_cnn_labeling.py to extract patches, train, and run inference")


if __name__ == "__main__":
    main()
