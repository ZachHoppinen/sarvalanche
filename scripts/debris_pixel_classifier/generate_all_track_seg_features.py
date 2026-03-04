"""
Generate CNN segmentation features for all tracks across all scenes.

Runs the debris segmenter in batches over every track in every scene,
extracts scalar features from the seg mask, and saves them alongside
existing track predictions for XGBoost retraining.

Usage
-----
    conda run -n sarvalanche python scripts/cnn/generate_seg_features.py

Output
------
    local/issw/seg_features.parquet  — one row per track, indexed by key
"""

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.track_classifier import _discover_run_files, zone_from_path
from sarvalanche.ml.debris_segmenter import DebrisSegmenter
from sarvalanche.ml.track_patch_extraction import (
    extract_context_patch,
    TRACK_MASK_CHANNEL,
)
from sarvalanche.ml.weight_utils import find_weights

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s – %(message)s')
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
OUTPUT_PATH  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/seg_features.parquet')
BATCH_SIZE   = 32
PATCH_SIZE   = 64

# ── Device + model ────────────────────────────────────────────────────────────

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
log.info('Device: %s', device)

model = DebrisSegmenter()
model.load_state_dict(torch.load(find_weights('debris_segmenter'), map_location=device))
model.eval().to(device)
log.info('Loaded segmenter from %s', find_weights('debris_segmenter'))


# ── Feature extraction from seg mask ─────────────────────────────────────────

def _positional_ratio(prob: np.ndarray, track_mask: np.ndarray, third: str) -> float:
    """Fraction of debris in top/middle/bottom third of track bbox."""
    h = prob.shape[0]
    slices = {
        'top':    slice(0,         h // 3),
        'middle': slice(h // 3,    2 * h // 3),
        'bottom': slice(2 * h // 3, h),
    }
    region      = prob[slices[third]]
    mask_region = track_mask[slices[third]]
    interior    = region[mask_region > 0.5]
    if len(interior) == 0:
        return 0.0
    return float((interior > 0.5).mean())


def _compactness(binary: np.ndarray) -> float:
    """Ratio of debris pixels to bounding box area — 1.0 = fills bbox, 0 = empty."""
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return 0.0
    h = binary.shape[0] - rows.argmax() - np.flip(rows).argmax()
    w = binary.shape[1] - cols.argmax() - np.flip(cols).argmax()
    bbox_area = max(1, h * w)
    return float(binary.sum() / bbox_area)


def seg_mask_features(prob: np.ndarray, track_mask: np.ndarray) -> dict:
    """Extract scalar XGBoost features from a (64, 64) seg probability map."""
    interior = prob[track_mask > 0.5]
    if len(interior) == 0:
        return {k: 0.0 for k in [
            'seg_debris_fraction', 'seg_mean_prob', 'seg_max_prob',
            'seg_p90_prob', 'seg_compactness',
            'seg_top_ratio', 'seg_mid_ratio', 'seg_bot_ratio',
        ]}

    binary = (prob > 0.5).astype(float)
    return {
        'seg_debris_fraction': float((interior > 0.5).mean()),
        'seg_mean_prob':       float(interior.mean()),
        'seg_max_prob':        float(interior.max()),
        'seg_p90_prob':        float(np.percentile(interior, 90)),
        'seg_compactness':     _compactness(binary),
        'seg_top_ratio':       _positional_ratio(prob, track_mask, 'top'),
        'seg_mid_ratio':       _positional_ratio(prob, track_mask, 'middle'),
        'seg_bot_ratio':       _positional_ratio(prob, track_mask, 'bottom'),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

all_rows: list[dict] = []
t_start = time.perf_counter()

for runs_dir in RUNS_DIRS:
    gpkg_paths, nc_paths = _discover_run_files(runs_dir)
    zone_to_gpkg = {zone_from_path(p): p for p in gpkg_paths}

    for nc_path in nc_paths:
        zone     = zone_from_path(nc_path)
        gpkg_path = zone_to_gpkg.get(zone)
        if gpkg_path is None:
            continue

        gdf = gpd.read_file(gpkg_path)
        ds  = load_netcdf_to_dataset(nc_path)
        date = nc_path.stem.split('_')[-1]  # last token is date
        zone_date = nc_path.stem

        log.info('Processing %s — %d tracks', nc_path.name, len(gdf))

        # Extract patches in batches
        indices  = list(gdf.index)
        patches_batch: list[np.ndarray] = []
        keys_batch:    list[str]        = []
        failed = 0

        def _flush_batch(patches_batch, keys_batch):
            """Run model on current batch, extract features, append to all_rows."""
            if not patches_batch:
                return
            x      = torch.from_numpy(np.stack(patches_batch)).float().to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()  # (B, 64, 64)

            for key, prob, patch in zip(keys_batch, probs, patches_batch):
                track_mask = patch[TRACK_MASK_CHANNEL]
                feats      = seg_mask_features(prob, track_mask)
                feats['key'] = key
                all_rows.append(feats)

        for idx in tqdm(indices, desc=zone_date, unit='trk'):
            row = gdf.loc[idx]
            key = f"{zone}|{date}|{idx}"

            try:
                patch = extract_context_patch(row, ds, size=PATCH_SIZE, src_crs=gdf.crs)
                patches_batch.append(patch)
                keys_batch.append(key)
            except Exception:
                log.debug('patch failed: %s', key, exc_info=True)
                failed += 1
                continue

            if len(patches_batch) >= BATCH_SIZE:
                _flush_batch(patches_batch, keys_batch)
                patches_batch = []
                keys_batch    = []

        # Flush remainder
        _flush_batch(patches_batch, keys_batch)

        if failed:
            log.warning('%s: %d tracks failed patch extraction', zone_date, failed)

        ds.close()

# ── Save ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(all_rows).set_index('key')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH)

elapsed = time.perf_counter() - t_start
log.info(
    'Done — %d tracks in %.1f min → %s',
    len(df), elapsed / 60, OUTPUT_PATH,
)
log.info('Features: %s', list(df.columns))
log.info('Sample:\n%s', df.head(3).to_string())