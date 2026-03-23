"""Extract v3 single-pair training patches from debris polygon labels.

Accepts multiple dates in a single invocation — loads netcdf and computes
pairs once, then extracts patches for each date.

Memory-efficient: computes one pair's SAR at a time (pairs-outer loop),
iterates all dates × positions for that pair, then frees.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/extract_patches.py \
        --nc season_*.nc \
        --date 2024-11-15 2024-12-29 2025-02-04 \
        --polygons labels_2024-11-15.gpkg labels_2024-12-29.gpkg labels_2025-02-04.gpkg \
        --out-dir patches/2024-11-15 patches/2024-12-29 patches/2025-02-04 \
        --geotiff-dir geotiffs/2024-11-15 geotiffs/2024-12-29 geotiffs/2025-02-04 \
        --val-paths akdot_paths.gpkg
"""

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rioxarray  # noqa: F401
from rasterio.transform import from_bounds
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_SAR, N_STATIC, STATIC_CHANNELS
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    extract_single_pair,
    get_pair_metadata_and_tracks,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SLOPE_IDX = STATIC_CHANNELS.index('slope')
DEM_IDX = STATIC_CHANNELS.index('dem')
CELL_COUNTS_IDX = STATIC_CHANNELS.index('cell_counts')


# ── Helpers ───────────────────────────────────────────────────────────

def rasterize_polygons(gdf, ds):
    H, W = ds.sizes["y"], ds.sizes["x"]
    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx/2, float(y.min()) - dy/2,
        float(x.max()) + dx/2, float(y.max()) + dy/2, W, H,
    )
    if gdf.crs and ds.rio.crs and gdf.crs != ds.rio.crs:
        gdf = gdf.to_crs(ds.rio.crs)
    if len(gdf) == 0:
        return np.zeros((H, W), dtype=np.float32)
    return rasterio.features.geometry_mask(
        gdf.geometry, out_shape=(H, W), transform=transform,
        invert=True, all_touched=True,
    ).astype(np.float32)


def rasterize_val_paths(val_paths_files, ds):
    H, W = ds.sizes["y"], ds.sizes["x"]
    mask = np.zeros((H, W), dtype=bool)
    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx/2, float(y.min()) - dy/2,
        float(x.max()) + dx/2, float(y.max()) + dy/2, W, H,
    )
    for vp in val_paths_files:
        if not vp.exists():
            continue
        gdf = gpd.read_file(vp)
        if gdf.crs and ds.rio.crs and gdf.crs != ds.rio.crs:
            gdf = gdf.to_crs(ds.rio.crs)
        if len(gdf) > 0:
            mask |= rasterio.features.geometry_mask(
                gdf.geometry, out_shape=(H, W), transform=transform,
                invert=True, all_touched=True,
            )
            log.info("  Val paths from %s: %d polygons, %d pixels",
                     vp.name, len(gdf), mask.sum())
    return mask


def build_reviewed_mask(geotiff_dir, ds):
    H, W = ds.sizes["y"], ds.sizes["x"]
    if geotiff_dir is None or not geotiff_dir.is_dir():
        return np.ones((H, W), dtype=bool)

    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx/2, float(y.min()) - dy/2,
        float(x.max()) + dx/2, float(y.max()) + dy/2, W, H,
    )
    mask = np.zeros((H, W), dtype=bool)
    tifs = list(geotiff_dir.glob("*.tif"))
    for tif in tifs:
        with rasterio.open(tif) as src:
            b = src.bounds
        fp_mask = rasterio.features.geometry_mask(
            [box(b.left, b.bottom, b.right, b.top)],
            out_shape=(H, W), transform=transform, invert=True,
        )
        mask |= fp_mask
    log.info("  Reviewed extent: %d geotiffs, %d pixels (%.1f%%)",
             len(tifs), mask.sum(), 100 * mask.mean())
    return mask


def compute_confidence(static_patch, label_patch, date_str):
    debris_px = label_patch > 0.5
    n_debris = int(debris_px.sum())
    if n_debris < 3:
        return 1.0

    slope = static_patch[SLOPE_IDX]
    dem = static_patch[DEM_IDX]
    cell_counts = static_patch[CELL_COUNTS_IDX]

    valid = np.isfinite(slope) & (slope > 0.01)
    n_valid = int(valid.sum())
    if n_valid < 10:
        return 1.0

    # Coverage
    coverage = n_debris / max(n_valid, 1)
    cov_conf = 0.1 if coverage > 0.40 else (0.3 if coverage > 0.25 else (0.6 if coverage > 0.15 else 1.0))

    # Elevation
    dem_debris = dem[debris_px]
    dem_valid = dem[valid]
    if np.isfinite(dem_debris).sum() > 2 and np.isfinite(dem_valid).sum() > 10:
        dem_offset = float(np.nanmean(dem_debris)) - float(np.nanmean(dem_valid))
        elev_conf = 0.5 if dem_offset > 0.15 else (0.7 if dem_offset > 0.05 else 1.0)
    else:
        elev_conf = 1.0

    # FlowPy overlap
    cc_debris = cell_counts[debris_px]
    if np.isfinite(cc_debris).sum() > 0:
        runout_frac = float((cc_debris > 0).sum()) / max(n_debris, 1)
        runout_conf = 1.0 if runout_frac > 0.3 else (0.8 if runout_frac > 0.1 else 0.5)
    else:
        runout_conf = 0.7

    # Spring penalty
    month = int(date_str[5:7])
    spring_conf = 0.3 if month >= 4 else (0.7 if month == 3 else 1.0)

    return float(min(cov_conf, elev_conf, runout_conf, spring_conf))


# ── Per-date config ───────────────────────────────────────────────────

class DateConfig:
    """Pre-computed data for one label date."""
    def __init__(self, date_str, debris_mask, reviewed_mask, val_path_mask,
                 static_scene, out_dir, pos_pair_indices, stride, neg_ratio,
                 min_debris_frac, patch_size):
        self.date_str = date_str
        self.debris_mask = debris_mask
        self.out_dir = out_dir
        self.pos_pair_indices = pos_pair_indices

        # Find positions within reviewed extent
        H, W = debris_mask.shape
        debris_positions = []
        nondebris_positions = []

        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                rev_frac = reviewed_mask[y0:y0 + patch_size, x0:x0 + patch_size].mean()
                if rev_frac < 0.5:
                    continue
                frac = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size].mean()
                if frac >= min_debris_frac:
                    debris_positions.append((y0, x0, frac))
                else:
                    nondebris_positions.append((y0, x0, 0.0))

        # Subsample negatives
        n_neg_max = int(len(debris_positions) * neg_ratio)
        if len(nondebris_positions) > n_neg_max:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(nondebris_positions), size=n_neg_max, replace=False)
            nondebris_positions = [nondebris_positions[i] for i in sorted(idx)]

        self.positions = [(True, y0, x0, f) for y0, x0, f in debris_positions] + \
                         [(False, y0, x0, f) for y0, x0, f in nondebris_positions]

        log.info("  %s: %d pos + %d neg = %d positions (stride=%d)",
                 date_str, len(debris_positions), len(nondebris_positions),
                 len(self.positions), stride)

        # Pre-compute static patches and confidence
        self.static_patches = {}
        self.label_patches = {}
        self.val_patches = {}
        self.confidences = {}
        self.metadata = {}
        zero_mask = np.zeros((patch_size, patch_size), dtype=np.float32)

        for has_debris, y0, x0, debris_frac in self.positions:
            pos_id = f"{'pos' if has_debris else 'neg'}_{y0:04d}_{x0:04d}"
            sp = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
            sp = normalize_dem_patch(sp)
            lp = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size]
            vp = val_path_mask[y0:y0 + patch_size, x0:x0 + patch_size]

            conf = compute_confidence(sp, lp, date_str) if has_debris else 1.0

            self.static_patches[pos_id] = sp
            self.label_patches[pos_id] = lp
            self.val_patches[pos_id] = vp
            self.confidences[pos_id] = conf
            self.metadata[pos_id] = {
                'label': 1 if has_debris else 0,
                'y0': int(y0), 'x0': int(x0),
                'debris_frac': float(debris_frac),
                'on_val_path': bool(vp.any()),
                'confidence': float(conf),
            }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract v3 single-pair training patches")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--date", type=str, nargs="+", required=True)
    parser.add_argument("--polygons", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--geotiff-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--val-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--min-debris-frac", type=float, default=0.005)
    args = parser.parse_args()

    n_dates = len(args.date)
    assert len(args.polygons) == n_dates, f"--polygons count ({len(args.polygons)}) != --date count ({n_dates})"
    assert len(args.out_dir) == n_dates, f"--out-dir count ({len(args.out_dir)}) != --date count ({n_dates})"

    # Pad geotiff-dir if not provided for all dates
    geotiff_dirs = list(args.geotiff_dir) + [None] * (n_dates - len(args.geotiff_dir))

    patch_size = V3_PATCH_SIZE

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    t_total = time.time()

    # ═══ SHARED WORK (once) ═══════════════════════════════════════════
    t0 = time.time()
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    H, W = ds.sizes["y"], ds.sizes["x"]
    log.info("  %d time steps, %d×%d", len(ds.time), H, W)

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        import xarray as xr
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR from %s", args.hrrr)

    log.info("  Dataset loaded in %.0fs", time.time() - t0)

    # Static stack (once)
    t0 = time.time()
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("  Static: %s", static_scene.shape)

    log.info("  Static stack built in %.0fs", time.time() - t0)

    # Val paths (once)
    val_path_mask = rasterize_val_paths(args.val_paths, ds)

    # Pair metadata + track data (once, no SAR materialization)
    t0 = time.time()
    log.info("Computing pair metadata...")
    pair_metas, tracks, hrrr_cache = get_pair_metadata_and_tracks(
        ds, max_span_days=args.max_span_days, hrrr_ds=hrrr_ds,
    )

    log.info("  Pair metadata computed in %.0fs", time.time() - t0)
    log.info("Shared setup done in %.0fs", time.time() - t_total)

    # ═══ PER-DATE SETUP ═══════════════════════════════════════════════
    t0 = time.time()
    date_configs = []
    for di in range(n_dates):
        date_str = args.date[di]
        out_dir = args.out_dir[di]
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Preparing %s...", date_str)

        # Load polygons
        gdf = gpd.read_file(args.polygons[di])
        log.info("  %d polygons", len(gdf))
        debris_mask = rasterize_polygons(gdf, ds)

        # Reviewed extent
        reviewed_mask = build_reviewed_mask(geotiff_dirs[di], ds)

        # Which pairs bracket this date?
        ref = pd.Timestamp(date_str)
        pos_pair_indices = set()
        for pi, meta in enumerate(pair_metas):
            if meta['t_start'] <= ref < meta['t_end']:
                pos_pair_indices.add(pi)
        log.info("  %d pairs bracket %s", len(pos_pair_indices), date_str)

        cfg = DateConfig(
            date_str, debris_mask, reviewed_mask, val_path_mask,
            static_scene, out_dir, pos_pair_indices,
            args.stride, args.neg_ratio, args.min_debris_frac, patch_size,
        )
        date_configs.append(cfg)

    total_positions = sum(len(cfg.positions) for cfg in date_configs)
    log.info("Total: %d positions across %d dates, %d pairs", total_positions, n_dates, len(pair_metas))
    log.info("Per-date setup done in %.0fs", time.time() - t0)

    # ═══ PAIRS-OUTER LOOP ═════════════════════════════════════════════
    # Compute one pair's SAR at a time, iterate all dates × positions, free
    t0 = time.time()
    n_saved = 0
    zero_mask = np.zeros((patch_size, patch_size), dtype=np.float32)

    pair_iter = enumerate(pair_metas)
    if HAS_TQDM:
        pair_iter = tqdm(list(pair_iter), desc="Pairs", unit="pair",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for pi, meta in pair_iter:

        # Compute SAR for this pair
        td = tracks[meta['track']]
        sar_scene = extract_single_pair(
            td['vv'], td['vh'], meta['i'], meta['j'], td['times'],
            td['anf'], hrrr_cache,
        )
        if sar_scene is None:
            continue

        # Coverage mask for this pair
        coverage = np.abs(sar_scene[0]) > 1e-6

        # Iterate all dates × positions for this pair
        for cfg in date_configs:
            is_bracketing = pi in cfg.pos_pair_indices

            for has_debris, y0, x0, debris_frac in cfg.positions:
                # Quick coverage check
                if not coverage[y0:y0 + patch_size, x0:x0 + patch_size].any():
                    continue

                pos_id = f"{'pos' if has_debris else 'neg'}_{y0:04d}_{x0:04d}"
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]

                # Per-pair label
                if has_debris and is_bracketing:
                    label_px = cfg.label_patches[pos_id]
                    # Check that covered debris pixels show backscatter increase.
                    # Use p98: at least some debris pixels must show strong positive
                    # change. Exclude zero-coverage pixels (different track).
                    has_coverage = np.abs(sar_patch[0]) > 1e-6
                    debris_covered = (label_px > 0.5) & has_coverage
                    debris_change = sar_patch[0][debris_covered]
                    if len(debris_change) > 0 and np.percentile(debris_change, 98) > 0.5:
                        pair_label = np.int8(1)
                        pair_mask = label_px.copy()
                    else:
                        # Pair brackets the date but debris zone shows no increase
                        # — treat as negative (don't teach noise = debris)
                        pair_label = np.int8(0)
                        pair_mask = zero_mask.copy()
                else:
                    pair_label = np.int8(0)
                    pair_mask = zero_mask.copy()

                # Zero out label where no SAR coverage
                no_cov = np.abs(sar_patch[0]) < 1e-6
                pair_mask[no_cov] = 0.0

                fname = cfg.out_dir / f"{pos_id}_v3_pair{pi:02d}.npz"
                np.savez_compressed(
                    fname,
                    sar=sar_patch,
                    static=cfg.static_patches[pos_id],
                    label=pair_label,
                    label_mask=pair_mask,
                    val_path_mask=cfg.val_patches[pos_id],
                    pair_track=meta['track'],
                    t_start=str(meta['t_start']),
                    t_end=str(meta['t_end']),
                    span_days=meta['span_days'],
                )
                n_saved += 1

        del sar_scene  # free immediately

        if HAS_TQDM and hasattr(pair_iter, 'set_postfix'):
            pair_iter.set_postfix(saved=n_saved)

    log.info("Pairs loop done in %.0fs (%d files saved)", time.time() - t0, n_saved)

    # ═══ SAVE METADATA ════════════════════════════════════════════════
    for cfg in date_configs:
        with open(cfg.out_dir / "labels.json", 'w') as f:
            json.dump(cfg.metadata, f, indent=2)
        n_pos = sum(1 for v in cfg.metadata.values() if v['label'] == 1)
        n_neg = sum(1 for v in cfg.metadata.values() if v['label'] == 0)
        n_val = sum(1 for v in cfg.metadata.values() if v.get('on_val_path'))
        log.info("  %s: %d pos + %d neg positions, %d on val paths",
                 cfg.date_str, n_pos, n_neg, n_val)

    log.info("Done. %d total files saved across %d dates in %.0fs (%.1f min).",
             n_saved, n_dates, time.time() - t_total, (time.time() - t_total) / 60)

    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
