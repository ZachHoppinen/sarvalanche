"""
Classify all tracks across all run directories using the trained XGBoost model.

Usage
-----
    conda run -n sarvalanche python scripts/track_classifier/classify_all_tracks.py

Outputs a JSON file mapping track keys to p_debris scores.
Processes one nc file at a time. For large zones, processes tracks in chunks
with GC between chunks to control peak memory.
Results are written incrementally so partial progress is preserved if interrupted.
"""

import gc
import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from sarvalanche.ml.track_classifier import TRACK_PREDICTOR_MODEL, _load_ds
from sarvalanche.ml.track_features import (
    STATIC_FEATURE_VARS,
    _PER_TRACK_GROUPS,
    compute_scene_context,
    extract_track_features,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s – %(message)s')
log = logging.getLogger(__name__)

CHUNK_SIZE = 500  # tracks per chunk (GC between chunks)

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/all_track_predictions.json')
PARTIAL_DIR = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/_partial_predictions')

# ── Load classifier ──────────────────────────────────────────────────────────
clf = joblib.load(TRACK_PREDICTOR_MODEL)
log.info("Loaded classifier from %s (%d features)", TRACK_PREDICTOR_MODEL, len(clf.feature_names_in_))

# ── Load existing labels for reference ────────────────────────────────────────
labels = {}
if LABELS_PATH.exists():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    log.info("Loaded %d existing labels", len(labels))


def _zone_from_stem(stem):
    return re.sub(r'_\d{4}-\d{2}-\d{2}$', '', stem)


def _date_from_stem(stem):
    m = re.search(r'(\d{4}-\d{2}-\d{2})$', stem)
    return m.group(1) if m else None


def _extract_one(row, ds, scene_ctx):
    try:
        return extract_track_features(row, ds, scene_ctx=scene_ctx)
    except Exception:
        return None


# ── Scene context vars (small subset, cheap to reproject) ────────────────────
_SCENE_CTX_VARS = {'d_empirical', 'combined_distance', 'slope', 'aspect'}


# ── Process each file independently ──────────────────────────────────────────
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

for runs_dir in RUNS_DIRS:
    danger = 'HIGH' if 'high_danger' in str(runs_dir) else 'LOW'

    zone_to_gpkg = {}
    for gpkg in sorted(runs_dir.glob('*.gpkg')):
        zone = _zone_from_stem(gpkg.stem)
        zone_to_gpkg[zone] = gpkg

    nc_files = sorted(runs_dir.glob('*.nc'))
    log.info("Processing %d nc files in %s [%s]", len(nc_files), runs_dir.name, danger)

    for nc_path in nc_files:
        zone = _zone_from_stem(nc_path.stem)
        date = _date_from_stem(nc_path.stem)
        gpkg_path = zone_to_gpkg.get(zone)
        partial_path = PARTIAL_DIR / f"{nc_path.stem}_{danger}.json"

        if gpkg_path is None:
            log.warning("No gpkg for zone %s, skipping %s", zone, nc_path.name)
            continue

        if partial_path.exists():
            log.info("Already processed %s, skipping", nc_path.name)
            continue

        gdf = gpd.read_file(gpkg_path)
        log.info("Scoring %s | %s [%s] — %d tracks", zone, date, danger, len(gdf))

        # Discover needed vars
        ds_peek = xr.open_dataset(nc_path)
        needed = set(STATIC_FEATURE_VARS) | {'slope', 'aspect'}
        for pattern in _PER_TRACK_GROUPS.values():
            needed |= {v for v in ds_peek.data_vars if pattern.match(v)}
        ds_peek.close()

        # Step 1: Compute scene context from a small subset of vars
        ds_ctx = _load_ds(nc_path, gdf.crs, var_whitelist=list(_SCENE_CTX_VARS))
        scene_ctx = compute_scene_context(ds_ctx)
        ds_ctx.close()
        del ds_ctx
        gc.collect()

        # Step 2: Load full needed vars for feature extraction
        ds = _load_ds(nc_path, gdf.crs, var_whitelist=list(needed))

        indices = list(gdf.index)
        file_results = {}

        # Process in chunks to control memory
        for chunk_start in range(0, len(indices), CHUNK_SIZE):
            chunk_indices = indices[chunk_start:chunk_start + CHUNK_SIZE]
            chunk_desc = f"{zone} | {date} [{chunk_start}:{chunk_start + len(chunk_indices)}]"

            feat_list = []
            for idx in tqdm(chunk_indices, desc=chunk_desc, unit="trk"):
                feat_list.append(_extract_one(gdf.loc[idx], ds, scene_ctx))

            # Batch predict this chunk
            valid_mask = [f is not None for f in feat_list]
            feature_dicts = [f if f is not None else {} for f in feat_list]

            X_chunk = pd.DataFrame(feature_dicts).reindex(columns=clf.feature_names_in_).fillna(0)
            probas = clf.predict_proba(X_chunk)[:, 1]

            for i, idx in enumerate(chunk_indices):
                key = f"{zone}|{date}|{idx}"
                entry = {
                    'p_debris': round(float(probas[i]), 4) if valid_mask[i] else None,
                    'danger': danger,
                }
                label_info = labels.get(key)
                if label_info is not None:
                    entry['label'] = label_info['label']
                file_results[key] = entry

            del feat_list, feature_dicts, X_chunk, probas
            gc.collect()

        # Write all results for this file
        with open(partial_path, 'w') as f:
            json.dump(file_results, f)
        log.info("  Wrote %d predictions to %s", len(file_results), partial_path.name)

        ds.close()
        del ds, gdf, file_results
        gc.collect()

# ── Merge all partial results ─────────────────────────────────────────────────
log.info("Merging partial results...")
merged = {}
for p in sorted(PARTIAL_DIR.glob('*.json')):
    with open(p) as f:
        merged.update(json.load(f))

with open(OUTPUT_PATH, 'w') as f:
    json.dump(merged, f, indent=2)
log.info("Saved %d track predictions to %s", len(merged), OUTPUT_PATH)

# Summary
n_total = len(merged)
n_scored = sum(1 for v in merged.values() if v['p_debris'] is not None)
n_debris = sum(1 for v in merged.values() if (v['p_debris'] or 0) >= 0.5)
log.info("Summary: %d total tracks, %d scored, %d (%.1f%%) predicted as debris (p >= 0.5)",
         n_total, n_scored, n_debris, 100 * n_debris / n_scored if n_scored > 0 else 0)
