"""
patch_labeler.py — Generate and label large (~10 km) scene patches for CNN training.

Computes p_empirical for a given date/tau, selects N viewing windows (half
high-prob, half low-prob based on mean p_empirical), and presents them for
binary labeling (debris / no debris).

On save, each labeled window is tiled into 64×64 CNN training patches with
stride = patch_size/4 and stored as .npz files.

Outputs:
  - patches/<date>/tile_YYYY_XXXX_SS.npz  — (C, 64, 64) CNN inputs
  - patches/<date>/labels.json            — {window_id: label} mapping

Usage:
    conda run -n sarvalanche python scripts/manual_labeling/patch_labeler.py \
        --nc 'local/issw/dual_tau_output/zone/season_dataset.nc' \
        --date 2025-02-04 \
        --tau 6 \
        --n-patches 100 \
        --patch-km 10
"""

import argparse
import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.track_patch_extraction import (
    N_PATCH_CHANNELS,
    _PATCH_DATA_VARS,
    _CHANNEL_NORM,
    _NORTHING_CH,
    _EASTING_CH,
    TRACK_MASK_CHANNEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

CNN_PATCH_SIZE = 64  # CNN training patch size (pixels)
V2_PATCH_SIZE = 128  # v2 CNN patch size

# Display layers for the labeling UI
DISPLAY_LAYERS = {
    'd_empirical':  {'cmap': 'RdBu_r',  'label': 'Empirical Dist', 'vmin': -2, 'vmax': 2},
    'p_empirical':  {'cmap': 'RdYlGn',  'label': 'P(empirical)',   'vmin': 0,  'vmax': 1},
    'slope':        {'cmap': 'bone',     'label': 'Slope',          'vmin': 0.26, 'vmax': 0.79},
    'cell_counts':  {'cmap': 'Blues',    'label': 'Cell Counts',    'vmin': 0,  'vmax': None},
}


# ---------------------------------------------------------------------------
# Compute empirical
# ---------------------------------------------------------------------------

def compute_empirical_for_date(ds, reference_date, tau_days):
    from sarvalanche.weights.temporal import get_temporal_weights
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )

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
        tau_days=tau_days,
    )
    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical
    return ds


# ---------------------------------------------------------------------------
# Normalization (matches CNN training)
# ---------------------------------------------------------------------------

def _normalize_channel(arr, var):
    cfg = _CHANNEL_NORM.get(var)
    if not cfg:
        return arr
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return arr


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------

def _get_resolution_m(ds):
    """Get approximate pixel resolution in meters from dataset coordinates."""
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        # Approximate meters at mid-latitude
        mid_lat = float(ds.y.values.mean())
        dx_m = dx * 111320 * np.cos(np.radians(mid_lat))
        dy_m = dy * 110540
        return (dx_m + dy_m) / 2
    return (dx + dy) / 2


# ---------------------------------------------------------------------------
# Window grid + selection
# ---------------------------------------------------------------------------

def build_window_grid(ds, window_px):
    """Build non-overlapping grid of (y0, x0) window origins."""
    H, W = ds.sizes['y'], ds.sizes['x']
    coords = []
    for y0 in range(0, H - window_px + 1, window_px):
        for x0 in range(0, W - window_px + 1, window_px):
            coords.append((y0, x0))
    return coords


def score_windows(ds, coords, window_px):
    """Compute mean p_empirical per window. Returns list of (y0, x0, mean_p)."""
    p = ds['p_empirical'].values
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

    # Interleave: high, low, high, low, ...
    selected = []
    for h, l in zip(high, low):
        selected.append((*h, 'high'))
        selected.append((*l, 'low'))
    if n_patches % 2 == 1 and len(scored_sorted) > n_patches:
        mid_idx = len(scored_sorted) // 2
        selected.append((*scored_sorted[mid_idx], 'mid'))

    return selected


# ---------------------------------------------------------------------------
# CNN tile extraction from a labeled window
# ---------------------------------------------------------------------------

def extract_cnn_tile(ds, y0, x0, patch_size=CNN_PATCH_SIZE):
    """Extract (C, patch_size, patch_size) CNN input from pixel coords."""
    H, W = ds.sizes['y'], ds.sizes['x']
    out = np.zeros((N_PATCH_CHANNELS, patch_size, patch_size), dtype=np.float32)

    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var in ds.data_vars:
            arr = np.nan_to_num(
                ds[var].values[y0:y0 + patch_size, x0:x0 + patch_size].astype(np.float32),
                nan=0.0,
            )
            out[ch] = _normalize_channel(arr, var)

    # Position channels
    ny = np.linspace(1 - 2 * y0 / H, 1 - 2 * (y0 + patch_size) / H, patch_size, dtype=np.float32)
    nx = np.linspace(-1 + 2 * x0 / W, -1 + 2 * (x0 + patch_size) / W, patch_size, dtype=np.float32)
    out[_NORTHING_CH] = ny[:, np.newaxis] * np.ones(patch_size, dtype=np.float32)
    out[_EASTING_CH] = np.ones(patch_size, dtype=np.float32)[:, np.newaxis] * nx

    out[TRACK_MASK_CHANNEL] = 1.0
    return out


def save_cnn_tiles(ds, y0, x0, window_px, label, out_dir):
    """Tile a labeled window into 64×64 CNN patches with stride=16 and save."""
    stride = CNN_PATCH_SIZE // 4  # 16 pixels
    tiles_saved = 0

    for ty in range(y0, y0 + window_px - CNN_PATCH_SIZE + 1, stride):
        for tx in range(x0, x0 + window_px - CNN_PATCH_SIZE + 1, stride):
            tile = extract_cnn_tile(ds, ty, tx)

            # Skip tiles that are mostly zero/NaN (outside data extent)
            data_channels = tile[:len(_PATCH_DATA_VARS)]
            if np.abs(data_channels).max() < 1e-6:
                continue

            tile_id = f"tile_{ty:04d}_{tx:04d}"
            geo = {
                'x_coords': ds.x.values[tx:tx + CNN_PATCH_SIZE],
                'y_coords': ds.y.values[ty:ty + CNN_PATCH_SIZE],
                'crs': str(ds.rio.crs),
            }
            np.savez_compressed(
                out_dir / f"{tile_id}.npz",
                patch=tile,
                label=np.int8(label),
                **geo,
            )
            tiles_saved += 1

    return tiles_saved


# ---------------------------------------------------------------------------
# V2 tile extraction (per-track/pol SAR + static terrain)
# ---------------------------------------------------------------------------

def save_v2_tiles(ds, y0, x0, window_px, label, out_dir, window_id):
    """Tile a labeled window into 128×128 v2 patches and save.

    Each .npz contains:
      - sar_maps: (N, 128, 128) per-track/pol backscatter change maps
      - static: (6, 128, 128) normalized static terrain channels
      - label: int8
    """
    from sarvalanche.ml.v2.patch_extraction import build_v2_patch, V2_PATCH_SIZE

    stride = V2_PATCH_SIZE // 4  # 32 pixels
    tiles_saved = 0

    for ty in range(y0, y0 + window_px - V2_PATCH_SIZE + 1, stride):
        for tx in range(x0, x0 + window_px - V2_PATCH_SIZE + 1, stride):
            sar_maps, static = build_v2_patch(ds, ty, tx, V2_PATCH_SIZE)

            # Skip patches with no signal
            if np.abs(sar_maps).max() < 1e-6:
                continue

            tile_id = f"{window_id}_v2_{ty:04d}_{tx:04d}"
            geo = {
                'x_coords': ds.x.values[tx:tx + V2_PATCH_SIZE],
                'y_coords': ds.y.values[ty:ty + V2_PATCH_SIZE],
                'crs': str(ds.rio.crs),
            }
            np.savez_compressed(
                out_dir / f"{tile_id}.npz",
                sar_maps=sar_maps,
                static=static,
                label=np.int8(label),
                **geo,
            )
            tiles_saved += 1

    return tiles_saved


# ---------------------------------------------------------------------------
# Labeling UI
# ---------------------------------------------------------------------------

def label_windows(ds, selected, window_px, out_dir, labels_path, v2=False):
    """Interactive matplotlib labeling loop."""
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
    else:
        labels = {}

    result = {'label': None}

    def on_key(event):
        if event.key == '1':
            result['label'] = 1  # debris present
            plt.close()
        elif event.key == '0':
            result['label'] = 0  # no debris
            plt.close()
        elif event.key == 'n':
            result['label'] = -1  # skip
            plt.close()
        elif event.key == 'q':
            result['label'] = 'quit'
            plt.close()
        elif event.key == 'left':
            result['label'] = 'back'
            plt.close()
        elif event.key == 'right':
            result['label'] = 'next'
            plt.close()

    n_pos = sum(1 for v in labels.values() if v.get('label') == 1)
    n_neg = sum(1 for v in labels.values() if v.get('label') == 0)
    total_tiles = sum(v.get('n_tiles', 0) for v in labels.values())
    log.info(
        "Existing labels: %d windows (%d debris, %d no-debris), %d CNN tiles",
        len(labels), n_pos, n_neg, total_tiles,
    )

    res_m = _get_resolution_m(ds)
    window_km = window_px * res_m / 1000

    i = 0
    while i < len(selected):
        y0, x0, mean_p, bucket = selected[i]
        window_id = f"window_{y0:04d}_{x0:04d}"

        existing = labels.get(window_id)
        existing_str = f"  [labeled: {existing['label']}]" if existing else ""

        display_vars = list(DISPLAY_LAYERS.keys())
        n_panels = len(display_vars)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f"{window_id}  |  {window_km:.1f}×{window_km:.1f} km  |  "
            f"mean_p={mean_p:.3f}  [{bucket}]{existing_str}  "
            f"({i + 1}/{len(selected)})  |  {n_pos} pos / {n_neg} neg  ({total_tiles} tiles)\n"
            f"1=debris  0=no-debris  n=skip  \u2190/\u2192=nav  q=quit",
            fontsize=10,
        )
        fig.canvas.mpl_connect('key_press_event', on_key)

        for ax, var in zip(axes, display_vars):
            opts = DISPLAY_LAYERS[var]
            if var in ds.data_vars:
                data = ds[var].values[y0:y0 + window_px, x0:x0 + window_px]
            else:
                data = np.zeros((window_px, window_px))

            im_kwargs = {'cmap': opts['cmap'], 'aspect': 'equal'}
            if opts['vmin'] is not None:
                im_kwargs['vmin'] = opts['vmin']
            if opts['vmax'] is not None:
                im_kwargs['vmax'] = opts['vmax']
            ax.imshow(data, **im_kwargs)
            ax.set_title(opts['label'], fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        result['label'] = None
        plt.show()

        if result['label'] == 'quit':
            break
        elif result['label'] == 'back':
            i = max(0, i - 1)
            continue
        elif result['label'] == 'next':
            i += 1
            continue
        elif result['label'] == -1:
            i += 1
            continue
        elif result['label'] is not None:
            lbl = result['label']

            # Save CNN tiles
            if v2:
                n_tiles = save_v2_tiles(ds, y0, x0, window_px, lbl, out_dir, window_id)
            else:
                n_tiles = save_cnn_tiles(ds, y0, x0, window_px, lbl, out_dir)

            labels[window_id] = {
                'label': lbl,
                'y0': y0,
                'x0': x0,
                'window_px': window_px,
                'mean_p': mean_p,
                'bucket': bucket,
                'n_tiles': n_tiles,
            }
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)

            if lbl == 1:
                n_pos += 1
            else:
                n_neg += 1
            total_tiles += n_tiles
            log.info(
                "%s → %s  (%d tiles saved, %d total)  "
                "(%d windows: %d debris / %d no-debris)",
                window_id, "DEBRIS" if lbl == 1 else "NO DEBRIS",
                n_tiles, total_tiles,
                len(labels), n_pos, n_neg,
            )
            i += 1

    log.info(
        "Done. %d windows labeled → %d CNN tiles, saved to %s",
        len(labels), total_tiles, labels_path,
    )


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------

def save_geotiff_windows(ds, selected, window_px, out_dir, date_str, tau):
    """Save d_empirical for each selected window as a georeferenced GeoTIFF.

    Also saves/appends patch footprints to a GeoPackage for tracking coverage.
    """
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
            "[%d/%d] Saved %s  (mean_p=%.3f, %s)",
            i + 1, len(selected), fname.name, mean_p, bucket,
        )

        # Build footprint polygon from window coordinates
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

    log.info("Saved %d GeoTIFFs to %s", len(selected), geotiff_dir)

    # Create labeling GeoPackage with a single tiny seed polygon so QGIS
    # recognises it as a polygon layer.
    labels_path = out_dir / f"avalanche_labels_{date_str}.gpkg"
    if not labels_path.exists():
        # Seed point at center of first patch, buffered to ~1 pixel
        y0, x0 = selected[0][0], selected[0][1]
        cx = float(ds.x.values[x0 + window_px // 2])
        cy = float(ds.y.values[y0 + window_px // 2])
        res = abs(float(ds.x.values[1] - ds.x.values[0]))
        seed = gpd.GeoDataFrame(
            geometry=[box(cx, cy, cx + res, cy + res)],
            crs=crs,
        )
        seed.to_file(labels_path, driver="GPKG")
        log.info("Created labeling GeoPackage with seed polygon: %s", labels_path)
    else:
        log.info("Labeling GeoPackage already exists, skipping: %s", labels_path)

    # Save/append patch footprints GeoPackage
    new_gdf = gpd.GeoDataFrame(footprints, crs=crs)
    footprints_path = out_dir / "patch_footprints.gpkg"
    if footprints_path.exists():
        existing = gpd.read_file(footprints_path)
        # Drop any existing rows for this date to avoid duplicates on re-run
        existing = existing[existing["date"] != date_str]
        combined = pd.concat([existing, new_gdf], ignore_index=True)
    else:
        combined = new_gdf
    combined.to_file(footprints_path, driver="GPKG")
    log.info("Patch footprints saved to %s (%d total rows)", footprints_path, len(combined))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Label large scene patches for CNN training")
    parser.add_argument("--nc", type=Path, required=True, help="Path to season_dataset.nc")
    parser.add_argument("--date", type=str, required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--tau", type=float, default=6, help="Temporal decay tau (default: 6)")
    parser.add_argument("--n-patches", type=int, default=100, help="Number of windows to label")
    parser.add_argument("--patch-km", type=float, default=10, help="Window size in km (default: 10)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <nc_dir>/patches/<date>)")
    parser.add_argument("--save-geotiff", action="store_true",
                        help="Save d_empirical windows as GeoTIFFs instead of interactive labeling")
    parser.add_argument("--v2", action="store_true",
                        help="Save 128×128 v2 format patches (per-track/pol SAR + static terrain)")
    args = parser.parse_args()

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        log.info("Loading into memory...")
        ds = ds.load()
    log.info("  %d time steps, %d×%d spatial", len(ds.time), ds.sizes['y'], ds.sizes['x'])

    # Compute window size in pixels
    res_m = _get_resolution_m(ds)
    window_px = int(round(args.patch_km * 1000 / res_m))
    # Round to multiple of patch size for clean tiling
    tile_sz = V2_PATCH_SIZE if args.v2 else CNN_PATCH_SIZE
    window_px = max(tile_sz, (window_px // tile_sz) * tile_sz)
    log.info(
        "Resolution: %.1f m/px → window = %d px (%.1f km)",
        res_m, window_px, window_px * res_m / 1000,
    )

    # Ensure w_resolution
    if "w_resolution" not in ds.data_vars:
        raise RuntimeError("w_resolution not found — re-run season pipeline first")

    # Compute empirical
    ref_date = pd.Timestamp(args.date)
    log.info("Computing empirical for %s (tau=%gd)...", args.date, args.tau)
    ds = compute_empirical_for_date(ds, ref_date, args.tau)

    # Build window grid and score
    log.info("Building window grid...")
    coords = build_window_grid(ds, window_px)
    scored = score_windows(ds, coords, window_px)
    log.info("  %d valid windows out of %d grid cells", len(scored), len(coords))

    if not scored:
        log.error("No valid windows found. Scene may be too small for %.0f km windows.", args.patch_km)
        return

    # Select high/low windows
    n_patches = min(args.n_patches, len(scored))
    selected = select_windows(scored, n_patches)
    log.info(
        "Selected %d windows: %d high, %d low",
        len(selected),
        sum(1 for *_, b in selected if b == 'high'),
        sum(1 for *_, b in selected if b == 'low'),
    )

    # Output paths
    out_dir = args.out_dir or (args.nc.parent / "patches" / args.date)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_geotiff:
        save_geotiff_windows(ds, selected, window_px, out_dir, args.date, args.tau)
    else:
        labels_path = out_dir / "labels.json"
        label_windows(ds, selected, window_px, out_dir, labels_path, v2=args.v2)


if __name__ == "__main__":
    main()
