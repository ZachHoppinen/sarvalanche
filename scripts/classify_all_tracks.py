"""
Classify all tracks across all run directories using the trained XGBoost model.

Usage
-----
    conda run -n sarvalanche python scripts/classify_all_tracks.py

Outputs a GeoPackage with all tracks and their p_debris scores.
Processes one file at a time with per-variable reprojection to stay within memory.
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

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/all_track_predictions.gpkg')
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
        partial_path = PARTIAL_DIR / f"{nc_path.stem}_{danger}.gpkg"

        if gpkg_path is None:
            log.warning("No gpkg for zone %s, skipping %s", zone, nc_path.name)
            continue

        # Skip if already processed
        if partial_path.exists():
            log.info("Already processed %s, skipping", nc_path.name)
            continue

        gdf = gpd.read_file(gpkg_path)
        log.info("Scoring %s | %s [%s] — %d tracks", zone, date, danger, len(gdf))

        # Discover needed vars
        ds_peek = xr.open_dataset(nc_path)
        needed = set(STATIC_FEATURE_VARS) | {'aspect'}
        for pattern in _PER_TRACK_GROUPS.values():
            needed |= {v for v in ds_peek.data_vars if pattern.match(v)}
        ds_peek.close()

        # Per-variable reprojection to control memory
        needed |= {'slope', 'aspect'}  # required for scene context
        ds = _load_ds(nc_path, gdf.crs, var_whitelist=list(needed))
        scene_ctx = compute_scene_context(ds)

        results = []
        for idx in tqdm(gdf.index, desc=f"{zone} | {date}", unit="trk"):
            row = gdf.loc[idx]
            try:
                feats = extract_track_features(row, ds, scene_ctx=scene_ctx)
                X_row = pd.DataFrame([feats]).reindex(columns=clf.feature_names_in_).fillna(0)
                p_debris = float(clf.predict_proba(X_row)[0, 1])
            except Exception as exc:
                log.debug("Failed on track %d: %s", idx, exc)
                p_debris = float('nan')

            label_key = f"{zone}|{date}|{idx}"
            label_info = labels.get(label_key, {})

            results.append({
                'zone': zone,
                'date': date,
                'danger': danger,
                'track_idx': int(idx),
                'p_debris': p_debris,
                'label': label_info.get('label'),
                'geometry': row.geometry,
            })

        # Write this file's results immediately
        file_gdf = gpd.GeoDataFrame(results, geometry='geometry', crs=gdf.crs)
        file_gdf.to_file(partial_path, driver='GPKG')
        log.info("  Wrote %d predictions to %s", len(file_gdf), partial_path.name)

        ds.close()
        del ds, results, file_gdf, gdf
        gc.collect()

# ── Merge all partial results ─────────────────────────────────────────────────
log.info("Merging partial results...")
partial_files = sorted(PARTIAL_DIR.glob('*.gpkg'))
if not partial_files:
    log.error("No partial results found!")
    exit(1)

parts = [gpd.read_file(p) for p in partial_files]
result_gdf = pd.concat(parts, ignore_index=True)
result_gdf = gpd.GeoDataFrame(result_gdf, geometry='geometry')
result_gdf.to_file(OUTPUT_PATH, driver='GPKG')
log.info("Saved %d track predictions to %s", len(result_gdf), OUTPUT_PATH)

# Summary
n_total = len(result_gdf)
n_debris = (result_gdf['p_debris'] >= 0.5).sum()
log.info("Summary: %d total tracks, %d (%.1f%%) predicted as debris (p >= 0.5)",
         n_total, n_debris, 100 * n_debris / n_total if n_total > 0 else 0)

summary = result_gdf.groupby(['danger', 'zone', 'date']).agg(
    n_tracks=('p_debris', 'count'),
    n_debris=('p_debris', lambda x: (x >= 0.5).sum()),
    mean_p=('p_debris', 'mean'),
).reset_index()
print("\nPer-zone/date summary:")
print(summary.to_string(index=False))
