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