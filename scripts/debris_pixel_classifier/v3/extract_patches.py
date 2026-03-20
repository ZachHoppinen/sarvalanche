"""Extract v3 single-pair training patches from debris polygon labels.

For each labeled date:
  1. Compute d_empirical + static stack
  2. Find all crossing pairs per track
  3. For each pair × each patch position → save one .npz file
  4. Mark patches overlapping AKDOT/AKRR validation paths

Output per pair per patch: {patch_id}_v3_pair{N}.npz containing:
  - sar: (N_SAR, 128, 128)
  - static: (N_STATIC, 128, 128)
  - label_mask: (128, 128)
  - label: int8
  - val_path_mask: (128, 128) bool — True on validation path pixels

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/extract_patches.py \
        --nc local/cnfaic/netcdfs/.../season_2025-2026_*.nc \
        --polygons local/cnfaic/debris_shapes/avalanche_labels_2025-12-15.gpkg \
        --date 2025-12-15 \
        --out-dir local/cnfaic/v3_experiment/patches/2025-12-15 \
        --hrrr local/cnfaic/hrrr_temperature_2526.nc \
        --val-paths local/cnfaic/reported/akdot/avy_path_frequency.gpkg
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rioxarray  # noqa: F401
from rasterio.transform import from_bounds
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_SAR, N_STATIC
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_season_pairs,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def compute_empirical_for_date(ds, reference_date, tau_days=6):
    """Compute per-track/pol empirical layers for a date."""
    import re
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)
    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_emp, d_emp = calculate_empirical_backscatter_probability(
        ds, ref_ts, use_agreement_boosting=True,
        agreement_strength=0.8, min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_emp
    ds["d_empirical"] = d_emp
    return ds


def rasterize_polygons(gdf, ds):
    """Rasterize debris polygons to dataset grid → (H, W) float32 mask."""
    H, W = ds.sizes["y"], ds.sizes["x"]
    x, y = ds.x.values, ds.y.values
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx / 2, float(y.min()) - dy / 2,
        float(x.max()) + dx / 2, float(y.max()) + dy / 2, W, H,
    )

    if gdf.crs and ds.rio.crs and gdf.crs != ds.rio.crs:
        gdf = gdf.to_crs(ds.rio.crs)

    if len(gdf) == 0:
        return np.zeros((H, W), dtype=np.float32)

    mask = rasterio.features.geometry_mask(
        gdf.geometry, out_shape=(H, W), transform=transform,
        invert=True, all_touched=True,
    ).astype(np.float32)
    return mask


def rasterize_val_paths(val_paths_gdf, ds):
    """Rasterize validation path polygons → (H, W) bool mask."""
    if val_paths_gdf is None or len(val_paths_gdf) == 0:
        return np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)

    H, W = ds.sizes["y"], ds.sizes["x"]
    x, y = ds.x.values, ds.y.values
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx / 2, float(y.min()) - dy / 2,
        float(x.max()) + dx / 2, float(y.max()) + dy / 2, W, H,
    )

    if val_paths_gdf.crs and ds.rio.crs and val_paths_gdf.crs != ds.rio.crs:
        val_paths_gdf = val_paths_gdf.to_crs(ds.rio.crs)

    mask = rasterio.features.geometry_mask(
        val_paths_gdf.geometry, out_shape=(H, W), transform=transform,
        invert=True, all_touched=True,
    )
    return mask


def main():
    parser = argparse.ArgumentParser(description="Extract v3 single-pair training patches")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=6)
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--val-paths", type=Path, nargs="+", default=[],
                        help="GeoPackage(s) with validation path polygons to hold out")
    parser.add_argument("--stride", type=int, default=128,
                        help="Patch stride (default 128 = no overlap, pairs provide variety)")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Neg:pos ratio per position (default 1.0, pairs multiply data)")
    parser.add_argument("--max-span-days", type=int, default=60,
                        help="Max pair span in days (default 60, no limit on pair count)")
    parser.add_argument("--max-pos-span-days", type=int, default=24,
                        help="Max span for a pair to get a positive label (default 24)")
    parser.add_argument("--min-debris-frac", type=float, default=0.005)
    args = parser.parse_args()

    patch_size = V3_PATCH_SIZE
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  %d time steps, %d×%d", len(ds.time), ds.sizes["y"], ds.sizes["x"])

    # Load HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        import xarray as xr
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR from %s", args.hrrr)

    # Compute empirical (needed for static stack)
    log.info("Computing empirical for %s...", args.date)
    ds = compute_empirical_for_date(ds, pd.Timestamp(args.date), args.tau)

    # Load and rasterize debris labels
    log.info("Loading polygons from %s", args.polygons)
    gdf = gpd.read_file(args.polygons)
    log.info("  %d polygons", len(gdf))
    debris_mask = rasterize_polygons(gdf, ds)
    log.info("  Debris pixels: %d (%.2f%%)", int(debris_mask.sum()), 100 * debris_mask.mean())

    # Load and rasterize validation paths
    val_path_mask = np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)
    for vp in args.val_paths:
        if vp.exists():
            vp_gdf = gpd.read_file(vp)
            val_path_mask |= rasterize_val_paths(vp_gdf, ds)
            log.info("  Val paths from %s: %d polygons, %d pixels",
                     vp.name, len(vp_gdf), val_path_mask.sum())

    # Build static stack
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("  Static: %s", static_scene.shape)

    # Get ALL season pairs (not just those bracketing the label date)
    log.info("Extracting all season pairs...")
    pairs = get_all_season_pairs(
        ds, max_span_days=args.max_span_days,
        hrrr_ds=hrrr_ds,
    )
    log.info("  %d total pairs", len(pairs))

    if not pairs:
        log.error("No crossing pairs found")
        return

    # Classify pairs: which ones bracket the avalanche date?
    ref = pd.Timestamp(args.date)
    pos_pair_indices = set()
    for pi, pair in enumerate(pairs):
        if pair['t_start'] <= ref < pair['t_end']:
            pos_pair_indices.add(pi)
    log.info("  %d pairs bracket %s", len(pos_pair_indices), args.date)

    # Find patch positions with debris
    H, W = ds.sizes["y"], ds.sizes["x"]
    debris_positions = []
    nondebris_positions = []

    for y0 in range(0, H - patch_size + 1, args.stride):
        for x0 in range(0, W - patch_size + 1, args.stride):
            patch_mask = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size]
            frac = patch_mask.mean()
            if frac >= args.min_debris_frac:
                debris_positions.append((y0, x0, frac))
            else:
                nondebris_positions.append((y0, x0, 0.0))

    # Subsample non-debris positions
    n_neg_max = int(len(debris_positions) * args.neg_ratio)
    if len(nondebris_positions) > n_neg_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(nondebris_positions), size=n_neg_max, replace=False)
        nondebris_positions = [nondebris_positions[i] for i in sorted(idx)]

    log.info("Positions: %d with debris, %d without", len(debris_positions), len(nondebris_positions))

    # Save patches with per-pair labels:
    #   - Debris position + tight-bracketing pair → positive (label=1, label_mask=debris)
    #   - Debris position + non-bracketing pair → negative (label=0, label_mask=0)
    #   - Non-debris position + any pair → negative (label=0, label_mask=0)
    metadata = {}
    n_saved = 0
    n_pos_saved = 0
    n_neg_saved = 0
    zero_mask = np.zeros((patch_size, patch_size), dtype=np.float32)

    all_positions = list(debris_positions) + list(nondebris_positions)
    is_debris_pos = [True] * len(debris_positions) + [False] * len(nondebris_positions)

    for pos_idx, (y0, x0, debris_frac) in enumerate(all_positions):
        has_debris = is_debris_pos[pos_idx]
        pos_id = f"{'pos' if has_debris else 'neg'}_{y0:04d}_{x0:04d}"

        # Slice static (shared across pairs)
        static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
        static_patch = normalize_dem_patch(static_patch)
        label_patch = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size]
        val_patch = val_path_mask[y0:y0 + patch_size, x0:x0 + patch_size]
        on_val_path = bool(val_patch.any())

        for pi, pair in enumerate(pairs):
            sar_scene = pair['sar']
            sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]

            # Skip if no SAR signal in this pair at this location
            if np.abs(sar_patch[0]).max() < 1e-6:
                continue

            # Per-pair label: positive only for debris positions with
            # bracketing pairs
            if has_debris and pi in pos_pair_indices:
                pair_label = np.int8(1)
                pair_mask = label_patch.copy()
            else:
                pair_label = np.int8(0)
                pair_mask = zero_mask.copy()

            # Zero out label wherever SAR has no coverage (no data → ignore)
            no_coverage = np.abs(sar_patch[0]) < 1e-6
            pair_mask[no_coverage] = 0.0

            fname = args.out_dir / f"{pos_id}_v3_pair{pi:02d}.npz"
            np.savez_compressed(
                fname,
                sar=sar_patch,
                static=static_patch,
                label=pair_label,
                label_mask=pair_mask,
                val_path_mask=val_patch,
                pair_track=pair['track'],
                t_start=str(pair['t_start']),
                t_end=str(pair['t_end']),
                span_days=pair['span_days'],
            )
            n_saved += 1
            if pair_label == 1:
                n_pos_saved += 1
            else:
                n_neg_saved += 1

        # Metadata uses the position-level label (for dataset split by val_path)
        metadata[pos_id] = {
            'label': 1 if has_debris else 0,
            'y0': int(y0),
            'x0': int(x0),
            'debris_frac': float(debris_frac),
            'on_val_path': on_val_path,
        }

    # Save metadata
    with open(args.out_dir / "labels.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    n_pos = sum(1 for v in metadata.values() if v['label'] == 1)
    n_neg = sum(1 for v in metadata.values() if v['label'] == 0)
    n_val = sum(1 for v in metadata.values() if v.get('on_val_path'))
    log.info("Saved %d pair-patch files (%d pos, %d neg patches, %d on val paths)",
             n_saved, n_pos, n_neg, n_val)

    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
