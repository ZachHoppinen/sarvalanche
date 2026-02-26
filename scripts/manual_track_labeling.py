import ast
import json
import random
import re
import threading
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from sarvalanche.ml.track_classifier import (
    TRACK_PREDICTOR_DIR,
    TRACK_PREDICTOR_MODEL,
    build_training_set,
    train_classifier,
)
from sarvalanche.ml.track_features import extract_track_features

RUNS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OBS_PATH    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/snfac_obs_2021_2025.csv')

layers = {
    'distance_mahalanobis': {'cmap': 'plasma',   'label': 'Mahalanobis Distance', 'vmin': 0.2, 'vmax': 0.7},
    'p_empirical':          {'cmap': 'RdYlGn_r', 'label': 'Empirical p-value',    'vmin': 0.2, 'vmax': 0.7},
    'slope':                {'cmap': 'bone',      'label': 'Slope (rad)',           'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45)},
    'cell_counts':          {'cmap': 'Blues',     'label': 'Cell Counts',           'vmin': None, 'vmax': None},
}


def parse_stem(stem):
    """Return (zone, date) from a filename stem like 'Banner_Summit_2025-02-04'."""
    m = re.search(r'^(.+)_(\d{4}-\d{2}-\d{2})$', stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def label_key(zone, date, track_idx):
    return f"{zone}|{date}|{track_idx}"


# ── Discover all (zone, date, gpkg, nc) combos ───────────────────────────────
all_items = []  # list of (zone, date, track_idx, gpkg_path, nc_path)
for gpkg in sorted(RUNS_DIR.glob('*.gpkg')):
    zone, date = parse_stem(gpkg.stem)
    if zone is None:
        print(f"Warning: could not parse zone/date from {gpkg.name}, skipping")
        continue
    nc = gpkg.with_suffix('.nc')
    if not nc.exists():
        print(f"Warning: no matching .nc for {gpkg.name}, skipping")
        continue
    gdf = gpd.read_file(gpkg)
    for idx in gdf.index:
        all_items.append((zone, date, idx, gpkg, nc))

# ── Load classifier (optional) ───────────────────────────────────────────────
clf = None
if TRACK_PREDICTOR_MODEL.exists():
    clf = joblib.load(TRACK_PREDICTOR_MODEL)
    print(f"Classifier loaded from {TRACK_PREDICTOR_MODEL}")
else:
    print(f"No classifier found at {TRACK_PREDICTOR_MODEL}, predictions disabled")

def _retrain_in_background():
    """Retrain XGBoost classifier on current labels and update the global clf."""
    global clf
    print(f"\n[retrain] Starting on {len(labels)} labels...")
    try:
        X, y = build_training_set(labels, RUNS_DIR)
        new_clf = train_classifier(X, y)
        TRACK_PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_clf, TRACK_PREDICTOR_MODEL)
        clf = new_clf
        print(f"[retrain] Done — model updated with {len(labels)} labels")
    except Exception as exc:
        print(f"[retrain] Failed: {exc}")

# ── Load observation CSV ──────────────────────────────────────────────────────
_obs_raw = pd.read_csv(OBS_PATH)
_obs_raw['date'] = pd.to_datetime(_obs_raw['date'])
_obs_raw = _obs_raw.dropna(subset=['location_point'])
_obs_raw['geometry'] = _obs_raw['location_point'].apply(
    lambda s: Point(ast.literal_eval(s)['lng'], ast.literal_eval(s)['lat'])
)
obs_gdf = gpd.GeoDataFrame(_obs_raw, geometry='geometry', crs='EPSG:4326')

# ── Load existing labels ──────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

# ── Filter already labeled, then shuffle within each file ────────────────────
unlabeled_all = [
    item for item in all_items
    if label_key(item[0], item[1], item[2]) not in labels
]

# Group by nc_path so reproject only fires once per file, not once per track
groups = defaultdict(list)
for item in unlabeled_all:
    groups[item[4]].append(item)

group_keys = list(groups.keys())
random.shuffle(group_keys)
for key in group_keys:
    random.shuffle(groups[key])
unlabeled = [item for key in group_keys for item in groups[key]]

print(f"{len(labels)} already labeled, {len(unlabeled_all)} remaining out of {len(all_items)} total")

# ── UI ───────────────────────────────────────────────────────────────────────
result = {'label': None}


def on_key(event):
    if event.key in ['0', '1', '2', '3']:
        result['label'] = int(event.key)
        plt.close()
    elif event.key == 'n':
        result['label'] = -1
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


# ── Main labeling loop ────────────────────────────────────────────────────────
buffer = 500
current_nc_path = None
ds = None
gdf = None

i = 0
while i < len(unlabeled):
    zone, date, idx, gpkg_path, nc_path = unlabeled[i]

    # Reload ds and gdf only when the source file changes
    if nc_path != current_nc_path:
        if ds is not None:
            ds.close()
        gdf = gpd.read_file(gpkg_path)
        _peek = xr.open_dataset(nc_path)
        _load_vars = list({
            *layers.keys(),
            *[v for v in _peek.data_vars if _peek[v].dims == ('y', 'x')],
        })
        _peek.close()
        ds = xr.open_dataset(nc_path)[_load_vars].astype(float).rio.write_crs('EPSG:4326')
        ds = ds.rio.reproject(gdf.crs)
        current_nc_path = nc_path

    path = gdf.loc[idx]
    bounds = path.geometry.bounds
    path_gdf = gpd.GeoDataFrame([path], geometry='geometry', crs=gdf.crs)

    clip_kwargs = dict(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer),
    )

    existing = labels.get(label_key(zone, date, idx))
    existing_str = f"  [labeled: {existing['label']}]" if existing else ""

    pred_str = ""
    if clf is not None:
        feats = extract_track_features(path, ds)
        X_row = pd.DataFrame([feats]).reindex(columns=clf.feature_names_in_).fillna(0)
        p_debris = float(clf.predict_proba(X_row)[0, 1])
        pred_str = f"  ▶ model: {p_debris:.0%} debris"

    avl_date = pd.Timestamp(date)
    nearby_obs = obs_gdf[
        (obs_gdf['date'] - avl_date).abs() <= pd.Timedelta(days=6)
    ]
    if not nearby_obs.empty:
        nearby_obs = nearby_obs.to_crs(gdf.crs)
    obs_str = f"  ({len(nearby_obs)} obs ±6d)" if not nearby_obs.empty else ""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{zone}  |  {date}  |  track {idx}{existing_str}{obs_str}  ({i + 1}/{len(unlabeled)}){pred_str}\n"
        f"0=high conf no   1=low conf no   2=low conf yes   3=high conf yes   n=unsure   ←/→=back/fwd   q=quit",
        fontsize=11,
    )
    fig.canvas.mpl_connect('key_press_event', on_key)

    for ax, (var, opts) in zip(axes.flat, layers.items()):
        plot_kwargs = dict(ax=ax, cmap=opts['cmap'], robust=True, add_colorbar=True)
        if opts['vmin'] is not None:
            plot_kwargs['vmin'] = opts['vmin']
        if opts['vmax'] is not None:
            plot_kwargs['vmax'] = opts['vmax']
        ds[var].sel(**clip_kwargs).plot(**plot_kwargs)
        path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
        if not nearby_obs.empty:
            nearby_obs.plot(ax=ax, color='cyan', markersize=6, marker='o',
                            edgecolors='black', linewidths=0.5, zorder=5)
        ax.set_title(opts['label'])
        ax.set_aspect('equal')

    plt.tight_layout()
    result['label'] = None
    plt.show()

    if result['label'] == 'quit':
        break
    elif result['label'] == 'back':
        i = max(0, i - 1)
    elif result['label'] == 'next':
        i += 1
    elif result['label'] is not None:
        key = label_key(zone, date, idx)
        labels[key] = {
            'label': result['label'],
            'zone': zone,
            'date': date,
            'track_idx': int(idx),
        }
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"{zone} | {date} | track {idx} → {result['label']}  ({len(labels)} labeled)")
        if len(labels) % 50 == 0:
            threading.Thread(target=_retrain_in_background, daemon=True).start()
            print(f"[retrain] Triggered in background ({len(labels)} labels)")
        i += 1

if ds is not None:
    ds.close()

print(f"Done. {len(labels)} paths labeled, saved to {OUTPUT_PATH}")
