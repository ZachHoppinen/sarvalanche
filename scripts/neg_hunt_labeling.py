"""Negative-hunting batch labeler.

Shows a grid of 9 tracks (3x3) sorted by lowest XGBoost probability.
Click tracks that ARE positive — everything else gets auto-labeled as negative (label=1).

Controls:
  - Click a track thumbnail to toggle it as POSITIVE (green border → will get label=2)
  - Press ENTER / SPACE to confirm: clicked = positive (2), unclicked = negative (1)
  - Press 's' to skip this entire batch (no labels saved)
  - Press 'q' to quit
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr

from sarvalanche.ml.track_classifier import TRACK_PREDICTOR_MODEL
from sarvalanche.ml.track_features import compute_scene_context, extract_track_features

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
SCORE_CACHE_PATH = RUNS_DIRS[0] / '_track_scores.json'

GRID_SIZE = 9  # 3x3 grid
BUFFER = 500   # meters around track polygon

# Which layer to show in the grid thumbnails
DISPLAY_VAR = 'combined_distance'
DISPLAY_OPTS = {'cmap': 'RdBu_r', 'vmin': 0, 'vmax': 3}


def parse_stem(stem):
    m = re.search(r'^(.+)_(\d{4}-\d{2}-\d{2})$', stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def label_key(zone, date, track_idx):
    return f"{zone}|{date}|{track_idx}"


def load_score_cache():
    if not SCORE_CACHE_PATH.exists():
        return {}
    try:
        with open(SCORE_CACHE_PATH) as f:
            raw = json.load(f)
        return {
            (parts[0], parts[1], int(parts[2])): v
            for k, v in raw.items()
            if len(parts := k.split('|')) == 3
        }
    except Exception as exc:
        print(f"[score] Failed to load cache: {exc}")
        return {}


# ── Discover all tracks ──────────────────────────────────────────────────────
all_items = []
for runs_dir in RUNS_DIRS:
    danger = 'HIGH' if 'high_danger' in str(runs_dir) else 'LOW'
    zone_to_gpkg = {}
    for gpkg in sorted(runs_dir.glob('*.gpkg')):
        gpkg_zone, _ = parse_stem(gpkg.stem)
        if gpkg_zone is not None:
            zone_to_gpkg[gpkg_zone] = gpkg
    for nc in sorted(runs_dir.glob('*.nc')):
        zone, date = parse_stem(nc.stem)
        if zone is None:
            continue
        gpkg = zone_to_gpkg.get(zone)
        if gpkg is None:
            continue
        gdf = gpd.read_file(gpkg)
        for idx in gdf.index:
            all_items.append((zone, date, idx, gpkg, nc, danger))

# Load existing labels, filter to unlabeled
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

unlabeled = [
    item for item in all_items
    if label_key(item[0], item[1], item[2]) not in labels
]

# Load scores and sort by ascending probability (most likely negatives first)
scores = load_score_cache()
if not scores:
    print("ERROR: No score cache found. Run manual_track_labeling.py (without --random) first to generate scores.")
    raise SystemExit(1)

scored_unlabeled = [
    item for item in unlabeled
    if (item[0], item[1], item[2]) in scores
]
scored_unlabeled.sort(key=lambda item: scores[(item[0], item[1], item[2])])

unscored = len(unlabeled) - len(scored_unlabeled)
print(f"{len(labels)} already labeled, {len(scored_unlabeled)} scored & unlabeled, {unscored} unscored (skipped)")

n_neg = sum(1 for v in labels.values() if v['label'] in (0, 1))
n_pos = sum(1 for v in labels.values() if v['label'] in (2, 3))
print(f"Current balance: {n_neg} neg / {n_pos} pos")

# ── Dataset cache ────────────────────────────────────────────────────────────
_ds_cache = {}  # nc_path → (ds, gdf)


def get_ds_gdf(gpkg_path, nc_path):
    if nc_path not in _ds_cache:
        gdf = gpd.read_file(gpkg_path)
        peek = xr.open_dataset(nc_path)
        load_vars = [v for v in peek.data_vars if peek[v].dims == ('y', 'x')]
        peek.close()
        ds = xr.open_dataset(nc_path)[load_vars].astype(float).rio.write_crs('EPSG:4326')
        ds = ds.rio.reproject(gdf.crs)
        _ds_cache[nc_path] = (ds, gdf)
    return _ds_cache[nc_path]


# ── Grid UI ──────────────────────────────────────────────────────────────────
batch_idx = 0
total_batches = (len(scored_unlabeled) + GRID_SIZE - 1) // GRID_SIZE

while batch_idx * GRID_SIZE < len(scored_unlabeled):
    batch = scored_unlabeled[batch_idx * GRID_SIZE : (batch_idx + 1) * GRID_SIZE]
    n_batch = len(batch)
    ncols = 3
    nrows = (n_batch + ncols - 1) // ncols

    # Track state: which cells are toggled positive
    toggled = [False] * n_batch
    border_patches = [None] * n_batch
    result = {'action': None}

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    # Hide unused axes
    for j in range(n_batch, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Plot each track thumbnail
    for j, item in enumerate(batch):
        zone, date, idx, gpkg_path, nc_path, danger = item
        ax = axes_flat[j]
        prob = scores[(zone, date, idx)]

        try:
            ds, gdf = get_ds_gdf(gpkg_path, nc_path)
            row = gdf.loc[idx]
            bounds = row.geometry.bounds
            path_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)

            clip_kwargs = dict(
                x=slice(bounds[0] - BUFFER, bounds[2] + BUFFER),
                y=slice(bounds[3] + BUFFER, bounds[1] - BUFFER),
            )
            da = ds[DISPLAY_VAR].sel(**clip_kwargs)
            da.plot(ax=ax, add_colorbar=False, **DISPLAY_OPTS)
            path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
        except Exception as exc:
            ax.text(0.5, 0.5, f'Error:\n{exc}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(f'{zone[:15]}.. {date}\ntrack {idx}  p={prob:.2f}', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add invisible border rectangle for toggling
        rect = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            fill=False, edgecolor='none', linewidth=5, zorder=100,
        )
        ax.add_patch(rect)
        border_patches[j] = rect

    fig.suptitle(
        f'Batch {batch_idx + 1}/{total_batches}  |  '
        f'Click = toggle POSITIVE  |  ENTER/SPACE = confirm  |  s = skip  |  q = quit\n'
        f'Unclicked tracks will be labeled NEGATIVE (1)',
        fontsize=12,
    )

    def on_click(event):
        if event.inaxes is None:
            return
        for j in range(n_batch):
            if event.inaxes == axes_flat[j]:
                toggled[j] = not toggled[j]
                border_patches[j].set_edgecolor('lime' if toggled[j] else 'none')
                border_patches[j].set_linewidth(6 if toggled[j] else 0)
                # Also update title to show selection
                zone, date, idx = batch[j][0], batch[j][1], batch[j][2]
                prob = scores[(zone, date, idx)]
                tag = ' [POS]' if toggled[j] else ''
                axes_flat[j].set_title(
                    f'{zone[:15]}.. {date}\ntrack {idx}  p={prob:.2f}{tag}',
                    fontsize=9, color='green' if toggled[j] else 'black',
                )
                fig.canvas.draw_idle()
                break

    def on_key(event):
        if event.key in ('enter', ' '):
            result['action'] = 'confirm'
            plt.close()
        elif event.key == 's':
            result['action'] = 'skip'
            plt.close()
        elif event.key == 'q':
            result['action'] = 'quit'
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

    if result['action'] == 'quit':
        break
    elif result['action'] == 'skip':
        batch_idx += 1
        continue
    elif result['action'] == 'confirm':
        n_new_neg = 0
        n_new_pos = 0
        for j, item in enumerate(batch):
            zone, date, idx = item[0], item[1], item[2]
            key = label_key(zone, date, idx)
            if toggled[j]:
                lbl = 2  # positive
                n_new_pos += 1
            else:
                lbl = 1  # negative
                n_new_neg += 1
            labels[key] = {
                'label': lbl,
                'zone': zone,
                'date': date,
                'track_idx': int(idx),
            }

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(labels, f, indent=2)

        total_neg = sum(1 for v in labels.values() if v['label'] in (0, 1))
        total_pos = sum(1 for v in labels.values() if v['label'] in (2, 3))
        print(f"Batch {batch_idx + 1}: +{n_new_neg} neg, +{n_new_pos} pos  "
              f"(total: {total_neg} neg / {total_pos} pos, {len(labels)} labeled)")
        batch_idx += 1

# Clean up cached datasets
for ds, _ in _ds_cache.values():
    ds.close()

total_neg = sum(1 for v in labels.values() if v['label'] in (0, 1))
total_pos = sum(1 for v in labels.values() if v['label'] in (2, 3))
print(f"\nDone. {len(labels)} labeled ({total_neg} neg / {total_pos} pos), saved to {OUTPUT_PATH}")
