"""CNN-assisted debris shape review tool.

Auto-generates candidate polygons from CNN segmentation output and presents
a matplotlib viewer for fast accept/reject review.

Usage::

    conda run -n sarvalanche python scripts/review_cnn_shapes.py --date 2023-03-13
    conda run -n sarvalanche python scripts/review_cnn_shapes.py --date 2023-03-13 --zone Banner_Summit
    conda run -n sarvalanche python scripts/review_cnn_shapes.py --date 2023-03-13 --threshold 0.3 --all
"""

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
import rasterio.features
import torch
import xarray as xr
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from sarvalanche.ml.track_features import (
    TRACK_MASK_CHANNEL,
    _patch_transform,
    aggregate_seg_features,
    extract_track_patch,
)
from sarvalanche.ml.track_patch_encoder import CNN_SEG_ENCODER_PATH, TrackSegEncoder

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CNN-assisted debris shape review tool')
parser.add_argument('--date', required=True, help='Avalanche date (YYYY-MM-DD)')
parser.add_argument('--zone', default=None, help='Filter to a specific zone')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Initial CNN probability threshold (default: 0.5)')
parser.add_argument('--min-seg-max', type=float, default=0.2,
                    help='Skip tracks where seg_max < this value (default: 0.2)')
parser.add_argument('--all', action='store_true', dest='include_labeled',
                    help='Include already-labeled tracks')

args = parser.parse_args()

# ── Paths ────────────────────────────────────────────────────────────────────
RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
SHAPES_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')

# Viewer panel config
PANEL_VARS = {
    'combined_distance': {'cmap': 'RdBu_r', 'label': 'Combined Distance', 'vmin': 0, 'vmax': 3},
    'd_empirical':       {'cmap': 'RdBu_r', 'label': 'Empirical Distance', 'vmin': -1.5, 'vmax': 1.5},
    'slope':             {'cmap': 'bone',    'label': 'Slope (rad)', 'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45)},
}

THRESHOLD_CYCLE = [0.3, 0.5, 0.7]


def parse_stem(stem):
    """Return (zone, date) from a filename stem like 'Banner_Summit_2025-02-04'."""
    m = re.search(r'^(.+)_(\d{4}-\d{2}-\d{2})$', stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def label_key(zone, date, track_idx):
    return f"{zone}|{date}|{track_idx}"


# ── Discover tracks matching --date ──────────────────────────────────────────
all_items = []  # (zone, date, track_idx, gpkg_path, nc_path)
for runs_dir in RUNS_DIRS:
    zone_to_gpkg = {}
    for gpkg in sorted(runs_dir.glob('*.gpkg')):
        gpkg_zone, _ = parse_stem(gpkg.stem)
        if gpkg_zone is not None:
            zone_to_gpkg[gpkg_zone] = gpkg
    for nc in sorted(runs_dir.glob('*.nc')):
        zone, date = parse_stem(nc.stem)
        if zone is None or date != args.date:
            continue
        if args.zone and zone != args.zone:
            continue
        gpkg = zone_to_gpkg.get(zone)
        if gpkg is None:
            continue
        gdf = gpd.read_file(gpkg)
        for idx in gdf.index:
            all_items.append((zone, date, idx, gpkg, nc))

print(f"Found {len(all_items)} tracks for date {args.date}"
      + (f" zone {args.zone}" if args.zone else ""))

if not all_items:
    print("No tracks found. Check --date and --zone arguments.")
    raise SystemExit(1)

# ── Load labels ──────────────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

# ── Load existing debris shapes ─────────────────────────────────────────────
if SHAPES_PATH.exists():
    debris_shapes = gpd.read_file(SHAPES_PATH)
    print(f"Loaded {len(debris_shapes)} existing debris shapes")
else:
    debris_shapes = gpd.GeoDataFrame(
        columns=['key', 'zone', 'date', 'track_idx', 'geometry'],
        geometry='geometry',
    )


def _save_shapes():
    if debris_shapes.crs is None and not debris_shapes.empty:
        print("[shapes] Warning: no CRS set, skipping save")
        return
    debris_shapes.to_file(SHAPES_PATH, driver='GPKG')


def _save_labels():
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(labels, f, indent=2)


# ── Load CNN seg encoder ────────────────────────────────────────────────────
seg_enc = None
if CNN_SEG_ENCODER_PATH.exists():
    try:
        # Detect input channels from saved weights
        state = torch.load(CNN_SEG_ENCODER_PATH, weights_only=True)
        in_ch = state['enc1.0.weight'].shape[1]
        seg_enc = TrackSegEncoder(in_channels=in_ch)
        seg_enc.load_state_dict(state)
        seg_enc.eval()
        _model_in_ch = in_ch
        print(f"Seg encoder loaded from {CNN_SEG_ENCODER_PATH} ({in_ch} channels)")
    except RuntimeError as exc:
        print(f"Seg encoder weights incompatible: {exc}")
        raise SystemExit(1)
else:
    print(f"No seg encoder found at {CNN_SEG_ENCODER_PATH}")
    raise SystemExit(1)


# ── Run CNN on all tracks, filter and sort ───────────────────────────────────
print("Running CNN inference on all tracks...")

track_data = []  # (zone, date, idx, gpkg, nc, seg_probs, seg_feats, patch)
current_nc_path = None
ds = None
gdf = None

for zone, date, idx, gpkg_path, nc_path in all_items:
    key = label_key(zone, date, idx)

    # Skip already labeled unless --all
    if not args.include_labeled and key in labels:
        continue

    # Load dataset when nc_path changes
    if nc_path != current_nc_path:
        if ds is not None:
            ds.close()
        gdf = gpd.read_file(gpkg_path)
        _peek = xr.open_dataset(nc_path)
        _load_vars = list({
            v for v in _peek.data_vars if _peek[v].dims == ('y', 'x')
        })
        _peek.close()
        ds = xr.open_dataset(nc_path)[_load_vars].astype(float).rio.write_crs('EPSG:4326')
        ds = ds.rio.reproject(gdf.crs)
        current_nc_path = nc_path

    row = gdf.loc[idx]

    try:
        patch = extract_track_patch(row, ds)
    except Exception as exc:
        print(f"  Skip {key}: patch extraction failed: {exc}")
        continue

    # Run CNN (slice channels if model was trained with fewer)
    with torch.no_grad():
        model_patch = patch[:_model_in_ch]
        x = torch.from_numpy(model_patch).unsqueeze(0)  # (1, C, H, W)
        logits = seg_enc.segment(x)                      # (1, 1, H, W)
        seg_probs = torch.sigmoid(logits).squeeze().numpy()  # (64, 64)

    track_mask = patch[TRACK_MASK_CHANNEL]
    seg_feats = aggregate_seg_features(seg_probs, track_mask)

    # Skip edge-of-raster tracks (lots of zero-fill in data channels)
    data_valid = np.count_nonzero(patch[0]) / patch[0].size  # combined_distance
    if data_valid < 0.7:
        continue

    # Filter low confidence
    if seg_feats['seg_max'] < args.min_seg_max:
        continue

    track_data.append((zone, date, idx, gpkg_path, nc_path, seg_probs, seg_feats, patch))

# Sort by seg_mean descending — rewards broad high-confidence areas
# rather than single hot pixels at raster edges
track_data.sort(key=lambda t: t[6]['seg_mean'], reverse=True)
print(f"{len(track_data)} tracks pass filters (seg_max >= {args.min_seg_max})")

if not track_data:
    print("No tracks to review.")
    raise SystemExit(0)


# ── Vectorization: seg_probs → CRS polygons ─────────────────────────────────
def _vectorize(seg_probs, threshold, ref_da, size=64):
    """Threshold seg_probs and vectorize to Shapely polygons in CRS coords.

    Returns list of Shapely polygons (may be empty).
    """
    binary = (seg_probs >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    transform = _patch_transform(ref_da, size)
    shapes_gen = rasterio.features.shapes(binary, mask=binary, transform=transform)
    polys = []
    for geom, val in shapes_gen:
        if val == 1:
            poly = shapely_shape(geom)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)

    if len(polys) > 1:
        merged = unary_union(polys)
        if merged.geom_type == 'Polygon':
            polys = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polys = list(merged.geoms)

    return polys


def _get_ref_da(row, ds_local, buffer=1000.0):
    """Get the reference DataArray clipped to track extent (for affine transform)."""
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    clip_sel = dict(
        x=slice(minx - buffer, maxx + buffer),
        y=slice(maxy + buffer, miny - buffer),
    )
    # Find a reference variable
    for var in ['combined_distance', 'd_empirical', 'slope']:
        if var in ds_local.data_vars:
            return ds_local[var].sel(**clip_sel)
    # Fallback to first 2D var
    for var in ds_local.data_vars:
        if ds_local[var].dims == ('y', 'x'):
            return ds_local[var].sel(**clip_sel)
    return None


# ── Viewer ───────────────────────────────────────────────────────────────────
result = {'action': None}
current_threshold_idx = [THRESHOLD_CYCLE.index(args.threshold)
                         if args.threshold in THRESHOLD_CYCLE else 1]


def _show_track(ti, threshold):
    """Display the 2x2 viewer for track at index ti. Returns (action, polygons)."""
    zone, date, idx, gpkg_path, nc_path, seg_probs, seg_feats, patch = track_data[ti]
    key = label_key(zone, date, idx)

    # Load dataset for this track
    gdf_local = gpd.read_file(gpkg_path)
    _peek = xr.open_dataset(nc_path)
    _load_vars = list({v for v in _peek.data_vars if _peek[v].dims == ('y', 'x')})
    _peek.close()
    ds_local = xr.open_dataset(nc_path)[_load_vars].astype(float).rio.write_crs('EPSG:4326')
    ds_local = ds_local.rio.reproject(gdf_local.crs)

    row = gdf_local.loc[idx]
    ref_da = _get_ref_da(row, ds_local)
    if ref_da is None:
        print(f"  No reference data for {key}, skipping")
        return 'next', []

    # Vectorize
    cnn_polys = _vectorize(seg_probs, threshold, ref_da)
    has_polys = len(cnn_polys) > 0

    # Clip bounds for raster display
    buffer = 1000.0
    bounds = row.geometry.bounds
    clip_sel = dict(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer),
    )
    path_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf_local.crs)

    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    title = (f"{zone}  |  {date}  |  track {idx}  |  "
             f"seg_mean={seg_feats['seg_mean']:.2f}  seg_max={seg_feats['seg_max']:.2f}  "
             f"thr={threshold}  ({ti + 1}/{len(track_data)})")
    subtitle = "y=accept  n=reject  t=cycle threshold  d=flag for QGIS  \u2190/\u2192=nav  q=quit"
    if not has_polys:
        subtitle = "NO CANDIDATES at this threshold  |  " + subtitle
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)

    # Helper to plot CNN polygons on an axes
    def _overlay_polys(ax, polys, color='lime', lw=2.5):
        for poly in polys:
            coords = list(poly.exterior.coords)
            p = MplPolygon(coords, closed=True, fill=False,
                           edgecolor=color, alpha=0.9,
                           linewidth=lw, linestyle='--', zorder=10)
            ax.add_patch(p)

    # Top-left: combined_distance + CNN polygon + track boundary
    ax = axes[0, 0]
    if 'combined_distance' in ds_local.data_vars:
        ds_local['combined_distance'].sel(**clip_sel).plot(
            ax=ax, cmap='RdBu_r', vmin=0, vmax=3, add_colorbar=True)
    path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
    _overlay_polys(ax, cnn_polys)
    ax.set_title('Combined Distance + CNN polygon', fontsize=9)
    ax.set_aspect('equal')

    # Top-right: d_empirical + CNN polygon + track boundary
    ax = axes[0, 1]
    if 'd_empirical' in ds_local.data_vars:
        ds_local['d_empirical'].sel(**clip_sel).plot(
            ax=ax, cmap='RdBu_r', vmin=-1.5, vmax=1.5, add_colorbar=True)
    path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
    _overlay_polys(ax, cnn_polys)
    ax.set_title('Empirical Distance + CNN polygon', fontsize=9)
    ax.set_aspect('equal')

    # Bottom-left: CNN seg_probs heatmap + track boundary (white)
    ax = axes[1, 0]
    # Show seg_probs in the patch pixel space with the affine transform
    transform = _patch_transform(ref_da, 64)
    extent = [transform.c, transform.c + transform.a * 64,
              transform.f + transform.e * 64, transform.f]
    ax.imshow(seg_probs, cmap='hot', vmin=0, vmax=1, extent=extent,
              origin='upper', aspect='equal')
    path_gdf.boundary.plot(ax=ax, color='white', linewidth=1.5)
    ax.set_title('CNN seg_probs heatmap', fontsize=9)
    ax.set_aspect('equal')

    # Bottom-right: slope + CNN polygon + track boundary
    ax = axes[1, 1]
    if 'slope' in ds_local.data_vars:
        ds_local['slope'].sel(**clip_sel).plot(
            ax=ax, cmap='bone', vmin=np.deg2rad(15), vmax=np.deg2rad(45),
            add_colorbar=True)
    path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
    _overlay_polys(ax, cnn_polys)
    ax.set_title('Slope + CNN polygon', fontsize=9)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Key handler
    action_result = {'action': None}

    def on_key(event):
        if event.key == 'y':
            if has_polys:
                action_result['action'] = 'accept'
            else:
                print("  No polygons to accept at this threshold. Try 't' to cycle.")
                return
            plt.close()
        elif event.key == 'n':
            action_result['action'] = 'reject'
            plt.close()
        elif event.key == 't':
            action_result['action'] = 'cycle_threshold'
            plt.close()
        elif event.key == 'd':
            action_result['action'] = 'flag_qgis'
            plt.close()
        elif event.key == 'left':
            action_result['action'] = 'prev'
            plt.close()
        elif event.key == 'right':
            action_result['action'] = 'next'
            plt.close()
        elif event.key == 'q':
            action_result['action'] = 'quit'
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    ds_local.close()
    return action_result['action'], cnn_polys


# ── Main review loop ────────────────────────────────────────────────────────
session_accepted = 0
session_rejected = 0
session_flagged = 0
i = 0
threshold = THRESHOLD_CYCLE[current_threshold_idx[0]]

while i < len(track_data):
    action, cnn_polys = _show_track(i, threshold)

    if action is None:
        # Window closed without key press
        i += 1
        continue

    zone, date, idx, gpkg_path, nc_path, seg_probs, seg_feats, patch = track_data[i]
    key = label_key(zone, date, idx)

    if action == 'quit':
        break
    elif action == 'prev':
        i = max(0, i - 1)
    elif action == 'next':
        i += 1
    elif action == 'cycle_threshold':
        current_threshold_idx[0] = (current_threshold_idx[0] + 1) % len(THRESHOLD_CYCLE)
        threshold = THRESHOLD_CYCLE[current_threshold_idx[0]]
        print(f"  Threshold → {threshold}")
        # Re-show same track, don't advance
    elif action == 'accept':
        # Save CNN polygons to debris_shapes.gpkg
        gdf_local = gpd.read_file(gpkg_path)

        # Remove existing shapes for this key (CNN replaces manual)
        if not debris_shapes.empty and 'key' in debris_shapes.columns:
            debris_shapes = debris_shapes[debris_shapes['key'] != key]
            debris_shapes = gpd.GeoDataFrame(debris_shapes, geometry='geometry')

        new_rows = gpd.GeoDataFrame(
            [{'key': key, 'zone': zone, 'date': date, 'track_idx': int(idx),
              'geometry': poly}
             for poly in cnn_polys],
            geometry='geometry',
            crs=gdf_local.crs,
        )
        debris_shapes = pd.concat([debris_shapes, new_rows], ignore_index=True)
        debris_shapes = gpd.GeoDataFrame(debris_shapes, geometry='geometry',
                                         crs=new_rows.crs if debris_shapes.crs is None else debris_shapes.crs)
        _save_shapes()

        # Label as 3 (positive, CNN-assisted)
        labels[key] = {
            'label': 3,
            'zone': zone,
            'date': date,
            'track_idx': int(idx),
        }
        _save_labels()
        session_accepted += 1
        print(f"  ACCEPT {key} → label=3, {len(cnn_polys)} polygon(s) saved "
              f"({len(debris_shapes)} total shapes)")
        i += 1
    elif action == 'reject':
        # Label as 1 (not debris)
        labels[key] = {
            'label': 1,
            'zone': zone,
            'date': date,
            'track_idx': int(idx),
        }
        _save_labels()
        session_rejected += 1
        print(f"  REJECT {key} → label=1")
        i += 1
    elif action == 'flag_qgis':
        # Flag for manual QGIS review
        labels[key] = {
            'label': -1,
            'zone': zone,
            'date': date,
            'track_idx': int(idx),
        }
        _save_labels()
        session_flagged += 1
        print(f"  FLAG {key} → label=-1 (for QGIS)")
        i += 1

# Summary
session_total = session_accepted + session_rejected + session_flagged
print(f"\nSession: {session_total} reviewed — {session_accepted} accepted, "
      f"{session_rejected} rejected, {session_flagged} flagged")
