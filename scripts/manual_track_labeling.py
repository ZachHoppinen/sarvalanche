import json
import random
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUNS_DIR = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')

layers = {
    'distance_mahalanobis': {'cmap': 'plasma',   'label': 'Mahalanobis Distance', 'vmin': 0.4, 'vmax': 1},
    'p_empirical':          {'cmap': 'RdYlGn_r', 'label': 'Empirical p-value',    'vmin': 0.4, 'vmax': 1},
    'slope':                {'cmap': 'bone',      'label': 'Slope (rad)',           'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45)},
    'cell_counts':          {'cmap': 'Blues',     'label': 'Cell Counts',           'vmin': 0,   'vmax': 1000},
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

# ── Load existing labels ──────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

# ── Filter already labeled, then shuffle ─────────────────────────────────────
unlabeled = [
    item for item in all_items
    if label_key(item[0], item[1], item[2]) not in labels
]
random.shuffle(unlabeled)
print(f"{len(labels)} already labeled, {len(unlabeled)} remaining out of {len(all_items)} total")

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


# ── Main labeling loop ────────────────────────────────────────────────────────
buffer = 500
current_nc_path = None
ds = None
gdf = None

for zone, date, idx, gpkg_path, nc_path in unlabeled:
    # Reload ds and gdf only when the source file changes
    if nc_path != current_nc_path:
        if ds is not None:
            ds.close()
        gdf = gpd.read_file(gpkg_path)
        ds = xr.open_dataset(nc_path).astype(float).rio.write_crs('EPSG:4326')
        ds = ds.rio.reproject(gdf.crs)
        current_nc_path = nc_path

    path = gdf.loc[idx]
    bounds = path.geometry.bounds
    path_gdf = gpd.GeoDataFrame([path], geometry='geometry', crs=gdf.crs)

    clip_kwargs = dict(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{zone}  |  {date}  |  track {idx}\n"
        f"0=high conf no   1=low conf no   2=low conf yes   3=high conf yes   n=unsure   q=quit",
        fontsize=11,
    )
    fig.canvas.mpl_connect('key_press_event', on_key)

    for ax, (var, opts) in zip(axes.flat, layers.items()):
        ds[var].sel(**clip_kwargs).plot(
            ax=ax, cmap=opts['cmap'], robust=True, add_colorbar=True,
            vmin=opts['vmin'], vmax=opts['vmax'],
        )
        path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
        ax.set_title(opts['label'])
        ax.set_aspect('equal')

    plt.tight_layout()
    result['label'] = None
    plt.show()

    if result['label'] == 'quit':
        break
    if result['label'] is not None:
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

if ds is not None:
    ds.close()

print(f"Done. {len(labels)} paths labeled, saved to {OUTPUT_PATH}")
