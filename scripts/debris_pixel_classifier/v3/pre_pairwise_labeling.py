"""Pre-labeling setup for pairwise debris CNN: export per-pair change GeoTIFFs for QGIS.

For each target date, finds SAR pairs where t_end is near that date, computes
the pairwise change channels (change_vv, change_vh, change_cr), selects ~20 km
windows with high/low signal, and exports multi-band GeoTIFFs + seed label gpkgs.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/pre_pairwise_labeling.py \
        --nc local/issw/uac/netcdfs/Logan/season_2024-2025_Logan.nc \
        --dates 2025-01-04 2025-02-15 \
        --out-dir local/issw/debris_shapes/uac

    # Or build season dataset on the fly:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/pre_pairwise_labeling.py \
        --zone Logan --center UAC --season 2024-2025 \
        --nc-dir local/issw/uac/netcdfs/ \
        --dates 2025-01-04 2025-02-15 \
        --out-dir local/issw/debris_shapes/uac
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import extract_all_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

N_PATCHES = 2
PATCH_KM = 20
MAX_SPAN_DAYS = 60


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

MIN_COVERAGE = 0.15  # pairs must cover at least 15% of the scene

def find_pairs_for_date(pair_metas, pair_diffs, target_date, max_offset_days=6):
    """Find pairs whose t_end is within max_offset_days of target_date.

    Filters out pairs whose valid mask covers less than MIN_COVERAGE of the
    scene (e.g. a track that barely clips the zone edge).

    Returns list of (pair_index, meta) sorted by offset then span_days.
    """
    target = pd.Timestamp(target_date)
    matches = []
    for i, meta in enumerate(pair_metas):
        offset = abs((meta["t_end"] - target).days)
        if offset <= max_offset_days:
            _, _, valid_mask = pair_diffs[i]
            coverage = float(valid_mask.mean())
            if coverage < MIN_COVERAGE:
                log.info("    Skipping pair %d (trk %s, %s->%s): %.1f%% coverage < %.0f%% threshold",
                         i, meta["track"], meta["t_start"].date(),
                         meta["t_end"].date(), coverage * 100, MIN_COVERAGE * 100)
                continue
            matches.append((i, meta, offset, coverage))

    if not matches:
        return []

    # Sort: prefer smallest offset, then shortest span
    matches.sort(key=lambda x: (x[2], x[1]["span_days"]))
    return [(i, m) for i, m, _, _ in matches]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def _get_resolution_m(ds):
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        mid_lat = float(ds.y.values.mean())
        dx_m = dx * 111320 * np.cos(np.radians(mid_lat))
        dy_m = dy * 110540
        return (dx_m + dy_m) / 2
    return (dx + dy) / 2


def build_window_grid(ny, nx, window_px):
    coords = []
    for y0 in range(0, ny - window_px + 1, window_px):
        for x0 in range(0, nx - window_px + 1, window_px):
            coords.append((y0, x0))
    return coords


def score_windows(change_vv, coords, window_px, urban_mask=None):
    """Score windows by mean positive change_vv, penalized by urban fraction."""
    scored = []
    for y0, x0 in coords:
        patch = change_vv[y0:y0 + window_px, x0:x0 + window_px]
        valid = np.isfinite(patch) & (patch != 0)
        frac_valid = float(valid.mean())
        if frac_valid < 0.3:
            continue
        # Score by mean of positive (brightening) signal — debris shows up as +VV
        pos_signal = np.where(patch > 0, patch, 0)
        mean_signal = float(np.mean(pos_signal[valid]))
        # Penalize windows with high urban fraction
        if urban_mask is not None:
            urban_patch = urban_mask[y0:y0 + window_px, x0:x0 + window_px]
            urban_frac = float(urban_patch.mean())
            mean_signal *= (1 - urban_frac)
        scored.append((y0, x0, mean_signal))
    return scored


def select_windows(scored, n_patches):
    """Select n_patches: half highest signal, half lowest."""
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
# GeoTIFF export
# ---------------------------------------------------------------------------

def export_pair_tiles(
    ds, change_vv, change_vh, anf_raw,
    selected, window_px, out_dir, pair_label, pair_meta,
):
    """Export 3-band GeoTIFFs (change_vv, change_vh, anf) + seed gpkg."""
    geotiff_dir = out_dir / "geotiffs" / pair_label
    geotiff_dir.mkdir(parents=True, exist_ok=True)

    crs = ds.rio.crs
    footprints = []

    for i, (y0, x0, mean_signal, bucket) in enumerate(selected):
        bands = np.stack([
            change_vv[y0:y0 + window_px, x0:x0 + window_px],
            change_vh[y0:y0 + window_px, x0:x0 + window_px],
            anf_raw[y0:y0 + window_px, x0:x0 + window_px],
        ], axis=0)

        y_vals = ds.y.values[y0:y0 + window_px]
        x_vals = ds.x.values[x0:x0 + window_px]

        da = xr.DataArray(
            bands,
            dims=("band", "y", "x"),
            coords={
                "band": ["change_vv", "change_vh", "anf"],
                "y": y_vals,
                "x": x_vals,
            },
        )
        da = da.rio.write_crs(crs)
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

        fname = geotiff_dir / f"pair_{pair_label}_{bucket}_{y0:04d}_{x0:04d}.tif"
        da.rio.to_raster(str(fname))

        log.info(
            "  [%d/%d] Saved %s  (mean_signal=%.3f, %s)",
            i + 1, len(selected), fname.name, mean_signal, bucket,
        )

        footprints.append({
            "geometry": box(
                float(x_vals.min()), float(y_vals.min()),
                float(x_vals.max()), float(y_vals.max()),
            ),
            "patch_id": f"{bucket}_{y0:04d}_{x0:04d}",
            "pair_label": pair_label,
            "mean_signal": mean_signal,
            "bucket": bucket,
            "geotiff": fname.name,
            "reviewed": False,
        })

    # Seed labeling GeoPackages (positive + negative)
    # Include track, t_start, t_end columns to match autolabel format
    y0, x0 = selected[0][0], selected[0][1]
    cx = float(ds.x.values[x0 + window_px // 2])
    cy = float(ds.y.values[y0 + window_px // 2])
    res = abs(float(ds.x.values[1] - ds.x.values[0]))
    seed = gpd.GeoDataFrame(
        {
            "geometry": [box(cx, cy, cx + res, cy + res)],
            "track": [pair_meta["track"]],
            "t_start": [pair_meta["t_start"].strftime("%Y-%m-%d")],
            "t_end": [pair_meta["t_end"].strftime("%Y-%m-%d")],
            "span_days": [pair_meta["span_days"]],
            "pair_label": [pair_label],
        },
        crs=crs,
    )

    labels_path = out_dir / f"avalanche_labels_{pair_label}.gpkg"
    if not labels_path.exists():
        seed.to_file(labels_path, driver="GPKG")
        log.info("  Created labeling GeoPackage: %s", labels_path)
    else:
        log.info("  Labeling GeoPackage already exists: %s", labels_path)

    # Append to patch_footprints.gpkg
    new_gdf = gpd.GeoDataFrame(footprints, crs=crs)
    footprints_path = out_dir / "patch_footprints.gpkg"
    if footprints_path.exists():
        existing = gpd.read_file(footprints_path)
        existing = existing[existing["pair_label"] != pair_label]
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
        description="Pre-labeling for pairwise debris CNN: export per-pair change GeoTIFFs",
    )

    nc_group = parser.add_mutually_exclusive_group(required=True)
    nc_group.add_argument("--nc", type=Path, help="Path to season dataset .nc")
    nc_group.add_argument("--zone", type=str, help="Zone name (triggers build_season_dataset)")

    parser.add_argument("--center", default="SNFAC", help="Avalanche center ID (with --zone)")
    parser.add_argument("--season", type=str, help='Winter season "YYYY-YYYY" (with --zone)')
    parser.add_argument("--nc-dir", type=Path, default=Path("local/issw/dual_tau_output/"))
    parser.add_argument("--dates", nargs="+", required=True, help="Target dates (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=Path, default=Path("local/issw/debris_shapes/uac"))
    parser.add_argument("--n-patches", type=int, default=N_PATCHES)
    parser.add_argument("--patch-km", type=float, default=PATCH_KM)
    parser.add_argument("--max-span-days", type=int, default=MAX_SPAN_DAYS)
    parser.add_argument("--max-offset-days", type=int, default=6,
                        help="Max days between target date and pair t_end")
    args = parser.parse_args()

    # ── Resolve season dataset ────────────────────────────────────────
    if args.zone:
        if not args.season:
            parser.error("--season required with --zone")

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "issw_analysis"))
        from build_season_dataset import (
            build_season_dataset, fetch_center_zones, season_nc_filename,
        )

        safe_name = args.zone.replace(" ", "_").replace("/", "-")
        zone_cache = args.nc_dir / safe_name
        nc_fname = season_nc_filename(args.season, args.zone)
        nc_path = zone_cache / nc_fname

        if not nc_path.exists():
            log.info("Building season dataset for %s %s...", args.zone, args.season)
            zones = fetch_center_zones(args.center)
            if args.zone not in zones:
                parser.error(f'Zone "{args.zone}" not found in {args.center}')
            start_year, end_year = args.season.split("-")
            build_season_dataset(
                aoi=zones[args.zone]["geometry"],
                season_start=f"{start_year}-11-01",
                season_end=f"{end_year}-04-30",
                cache_dir=zone_cache,
                nc_filename=nc_fname,
            )
    else:
        nc_path = args.nc

    if not nc_path.exists():
        log.error("Season dataset not found: %s", nc_path)
        return

    # ── Load dataset ──────────────────────────────────────────────────
    log.info("Loading %s", nc_path)
    ds = load_netcdf_to_dataset(nc_path)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  %d time steps, %d x %d spatial", len(ds.time), ds.sizes["y"], ds.sizes["x"])

    # ── Extract all pairs ─────────────────────────────────────────────
    log.info("Extracting pairs (max_span=%d days)...", args.max_span_days)
    pair_diffs, pair_metas, _, anf_raw_per_track = extract_all_pairs(
        ds, max_span_days=args.max_span_days,
    )

    # ── Compute window size ───────────────────────────────────────────
    PATCH_SIZE = 128
    res_m = _get_resolution_m(ds)
    window_px = int(round(args.patch_km * 1000 / res_m))
    window_px = max(PATCH_SIZE, (window_px // PATCH_SIZE) * PATCH_SIZE)
    log.info("Resolution: %.1f m/px -> window = %d px (%.1f km)",
             res_m, window_px, window_px * res_m / 1000)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each date ─────────────────────────────────────────────
    total_windows = 0
    for date_str in args.dates:
        log.info("=== Date: %s ===", date_str)

        matches = find_pairs_for_date(
            pair_metas, pair_diffs, date_str, max_offset_days=args.max_offset_days,
        )
        if not matches:
            log.warning("  No pairs found within %d days of %s", args.max_offset_days, date_str)
            continue

        log.info("  Found %d matching pairs", len(matches))
        for idx, meta in matches[:5]:
            log.info("    pair %d: trk %s, %s -> %s (%dd)",
                     idx, meta["track"], meta["t_start"].date(),
                     meta["t_end"].date(), meta["span_days"])

        # Use the best (closest t_end, shortest span) pair.
        # Verify t_start and t_end are from the same track (orbital path).
        best_idx, best_meta = matches[0]
        if best_meta["t_start"] == best_meta["t_end"]:
            raise ValueError(
                f"Pair {best_idx} has identical t_start and t_end "
                f"({best_meta['t_start']}). Cannot compute change."
            )
        # extract_all_pairs guarantees both timestamps are from the same
        # track, but verify explicitly.
        track_str = best_meta["track"]
        if not track_str:
            raise ValueError(
                f"Pair {best_idx} ({best_meta['t_start'].date()} -> "
                f"{best_meta['t_end'].date()}) has no track assigned. "
                "All pairs must be within the same orbital track."
            )
        vv_diff, vh_diff, valid_mask = pair_diffs[best_idx]

        pair_label = (
            f"trk{track_str}_{best_meta['t_start'].strftime('%Y-%m-%d')}"
            f"_{best_meta['t_end'].strftime('%Y-%m-%d')}"
            f"_{best_meta['span_days']}d"
        )
        log.info("  Selected pair: %s", pair_label)

        # Raw dB diffs — no normalization, keep interpretable for labeling
        change_vv = vv_diff.copy()
        change_vh = vh_diff.copy()
        anf_raw = anf_raw_per_track.get(track_str, np.ones_like(vv_diff))

        # Mask invalid pixels, water, and urban/infrastructure
        mask = valid_mask.copy()
        if "water_mask" in ds.data_vars:
            water = ds["water_mask"].values.astype(bool)
            mask = mask & ~water
            log.info("  Masked %d water pixels (%.1f%%)",
                     water.sum(), water.sum() / water.size * 100)
        if "urban_mask" in ds.data_vars:
            urban = ds["urban_mask"].values.astype(bool)
            mask = mask & ~urban
            log.info("  Masked %d urban pixels (%.1f%%)",
                     urban.sum(), urban.sum() / urban.size * 100)
        change_vv[~mask] = 0
        change_vh[~mask] = 0

        # Build window grid and score
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        coords = build_window_grid(ny, nx, window_px)
        urban = ds["urban_mask"].values.astype(bool) if "urban_mask" in ds.data_vars else None
        scored = score_windows(change_vv, coords, window_px, urban_mask=urban)
        log.info("  %d valid windows out of %d grid cells", len(scored), len(coords))

        if not scored:
            log.warning("  No valid windows for %s, skipping", date_str)
            continue

        n = min(args.n_patches, len(scored))
        selected = select_windows(scored, n)
        log.info("  Selected %d windows: %d high, %d low",
                 len(selected),
                 sum(1 for *_, b in selected if b == "high"),
                 sum(1 for *_, b in selected if b == "low"))

        n_saved = export_pair_tiles(
            ds, change_vv, change_vh, anf_raw,
            selected, window_px, args.out_dir, pair_label, best_meta,
        )
        total_windows += n_saved

    log.info(
        "Done. Exported %d windows across %d dates to %s",
        total_windows, len(args.dates), args.out_dir,
    )
    log.info("")
    log.info("Next steps:")
    log.info("  1. Open change_vv GeoTIFFs in QGIS: %s/geotiffs/", args.out_dir)
    log.info("     (Bright = VV brightening = potential debris)")
    log.info("  2. Draw debris polygons in: %s/avalanche_labels_<pair>.gpkg", args.out_dir)
    log.info("  3. Run post-labeling to extract training patches")


if __name__ == "__main__":
    main()
