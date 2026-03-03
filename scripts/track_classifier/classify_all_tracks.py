"""
Classify all tracks across all run directories using the trained XGBoost model.

Usage
-----
    conda run -n sarvalanche python scripts/track_classifier/classify_all_tracks.py

Outputs a JSON file mapping track keys to p_debris scores.
Processes one nc file at a time, writing partial results incrementally so
progress is preserved if interrupted.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.track_classifier import TRACK_PREDICTOR_MODEL, predict_tracks, _discover_run_files, zone_from_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s – %(message)s')
log = logging.getLogger(__name__)

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
LABELS_PATH  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OUTPUT_PATH  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/all_track_predictions.json')
PARTIAL_DIR  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/_partial_predictions')

# ── Load classifier ───────────────────────────────────────────────────────────
clf = joblib.load(TRACK_PREDICTOR_MODEL)
log.info("Loaded classifier from %s (%d features)", TRACK_PREDICTOR_MODEL, len(clf.feature_names_in_))

# ── Load labels for reference ─────────────────────────────────────────────────
labels = {}
if LABELS_PATH.exists():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    log.info("Loaded %d existing labels", len(labels))

# ── Process each file ─────────────────────────────────────────────────────────
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

for runs_dir in RUNS_DIRS:
    danger = 'HIGH' if 'high_danger' in str(runs_dir) else 'LOW'
    gpkg_paths, nc_paths = _discover_run_files(runs_dir)
    log.info("Processing %d nc files in %s [%s]", len(nc_paths), runs_dir.name, danger)

    for gpkg_path, nc_path in zip(gpkg_paths, nc_paths):
        zone = zone_from_path(nc_path)
        date = nc_path.stem.split('_')[-1]
        partial_path = PARTIAL_DIR / f"{nc_path.stem}_{danger}.json"

        if partial_path.exists():
            log.info("Already processed %s, skipping", nc_path.name)
            continue

        gdf = gpd.read_file(gpkg_path)
        log.info("Scoring %s | %s [%s] — %d tracks", zone, date, danger, len(gdf))

        ds = load_netcdf_to_dataset(nc_path)
        result_gdf = predict_tracks(clf, gdf, ds, n_jobs=1)
        ds.close()

        file_results = {}
        for idx, row in result_gdf.iterrows():
            key = f"{zone}|{date}|{idx}"
            entry = {'p_debris': round(float(row['p_debris']), 4), 'danger': danger}
            if key in labels:
                entry['label'] = labels[key]['label']
            file_results[key] = entry

        with open(partial_path, 'w') as f:
            json.dump(file_results, f)
        log.info("Wrote %d predictions to %s", len(file_results), partial_path.name)
# ── Merge partial results ─────────────────────────────────────────────────────
log.info("Merging partial results...")
merged = {}
for p in sorted(PARTIAL_DIR.glob('*.json')):
    with open(p) as f:
        merged.update(json.load(f))

with open(OUTPUT_PATH, 'w') as f:
    json.dump(merged, f, indent=2)
log.info("Saved %d track predictions to %s", len(merged), OUTPUT_PATH)

n_scored = sum(1 for v in merged.values() if v['p_debris'] is not None)
n_debris = sum(1 for v in merged.values() if (v['p_debris'] or 0) >= 0.5)
log.info(
    "Summary: %d total, %d scored, %d (%.1f%%) predicted debris (p >= 0.5)",
    len(merged), n_scored, n_debris,
    100 * n_debris / n_scored if n_scored else 0,
)

import json
from pathlib import Path
from collections import defaultdict

with open(OUTPUT_PATH) as f:
    preds = json.load(f)

# Break down by danger level
by_danger = defaultdict(list)
for v in preds.values():
    by_danger[v['danger']].append(v['p_debris'])

for danger, scores in sorted(by_danger.items()):
    scores = [s for s in scores if s is not None]
    above = sum(1 for s in scores if s >= 0.5)
    print(f"{danger}  n={len(scores):6d}  debris={above/len(scores):.1%}  "
          f"mean_p={sum(scores)/len(scores):.3f}")

# Distribution of scores
import numpy as np
all_scores = [v['p_debris'] for v in preds.values() if v['p_debris'] is not None]
for thresh in (0.3, 0.5, 0.7, 0.9):
    n = sum(1 for s in all_scores if s >= thresh)
    print(f"p >= {thresh}: {n:6d} ({100*n/len(all_scores):.1f}%)")

# Check labeled tracks — how well does p_debris align with labels?
labeled = {k: v for k, v in preds.items() if 'label' in v}
if labeled:
    tp = sum(1 for v in labeled.values() if v['p_debris'] >= 0.5 and v['label'] >= 2)
    tn = sum(1 for v in labeled.values() if v['p_debris'] <  0.5 and v['label'] <  2)
    fp = sum(1 for v in labeled.values() if v['p_debris'] >= 0.5 and v['label'] <  2)
    fn = sum(1 for v in labeled.values() if v['p_debris'] <  0.5 and v['label'] >= 2)
    print(f"\nLabeled tracks ({len(labeled)}):")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Precision={tp/(tp+fp):.3f}  Recall={tp/(tp+fn):.3f}")


import json
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.track_classifier import _discover_run_files, zone_from_path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

from shapely.geometry import MultiPolygon

def plot_geom_outline(ax, geom, **kwargs):
    """Plot polygon or multipolygon outline on ax."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        coords = np.array(poly.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, fill=False, **kwargs))

OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/all_track_predictions.json')
RUNS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs')

with open(OUTPUT_PATH) as f:
    preds = json.load(f)

# Find high-confidence unlabeled tracks
unlabeled_high = [
    (k, v) for k, v in preds.items()
    if 'label' not in v and v['p_debris'] >= 0.9
]
print(f"High confidence unlabeled tracks: {len(unlabeled_high)}")

# Pick a scene and plot a sample
gpkg_paths, nc_paths = _discover_run_files(RUNS_DIR)
sample_nc   = nc_paths[0]
sample_gpkg = gpkg_paths[0]
stem = sample_nc.stem
zone = zone_from_path(sample_nc)
date = stem.split('_')[-1]

scene_keys = {k: v for k, v in unlabeled_high
              if k.startswith(f"{zone}|{date}|")}
print(f"Scene {stem}: {len(scene_keys)} high-confidence unlabeled tracks")
import os
os.chdir('/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures/track_classifications')
if scene_keys:
    gdf = gpd.read_file(sample_gpkg)
    ds  = load_netcdf_to_dataset(sample_nc)

    sample_keys = list(scene_keys.items())[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for ax, (key, meta) in zip(axes.flat, sample_keys):
        idx = int(key.split('|')[-1])
        if idx not in gdf.index:
            continue

        geom = reproject_geom(gdf.loc[idx].geometry, gdf.crs, ds.rio.crs)
        minx, miny, maxx, maxy = geom.bounds
        buf = 0.009

        from sarvalanche.utils.raster_utils import _y_slice
        da = ds['d_empirical'].sel(
            x=slice(minx - buf, maxx + buf),
            y=_y_slice(ds['d_empirical'], miny - buf, maxy + buf)
        )
        arr = da.values
        x_vals = da.x.values
        y_vals = da.y.values

        ax.imshow(
            arr, cmap='RdBu_r', vmin=-2, vmax=2,
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            origin='upper', aspect='auto',
        )

        # Overlay track polygon
        plot_geom_outline(ax, geom, edgecolor='yellow', linewidth=1.5)

        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.set_ylim(y_vals.min(), y_vals.max())
        ax.set_title(f"p={meta['p_debris']:.3f}  idx={idx}", fontsize=8)
        ax.axis('off')

    plt.suptitle(f'{stem} — high confidence unlabeled debris', fontsize=11)
    plt.tight_layout()
    plt.savefig('unlabeled_high_conf.png', dpi=150)
    plt.show()
    ds.close()
