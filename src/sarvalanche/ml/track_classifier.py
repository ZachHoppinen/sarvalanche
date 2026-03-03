"""
XGBoost-based avalanche debris classifier.

Covers the full supervised learning pipeline for track-level debris detection:

  - build_training_set  : iterate labeled tracks, extract features, return X/y
  - train_classifier    : fit an XGBClassifier on the resulting feature matrix
  - predict_tracks      : score all tracks in a GeoDataFrame, returning p_debris

Feature extraction delegates entirely to sarvalanche.ml.track_features.
File discovery and loading delegate to sarvalanche.io.load_data.
Track iteration delegates to sarvalanche.utils.generators.
"""

import logging
import warnings
from pathlib import Path
import re

warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier

from sarvalanche.ml.track_features import (
    STATIC_FEATURE_VARS,
    _PER_TRACK_GROUPS,
    compute_scene_context,
    extract_track_features,
)
from sarvalanche.utils.generators import iter_labeled_run_tracks

from sarvalanche.ml.weight_utils import find_weights

TRACK_PREDICTOR_DIR: Path = Path(__file__).parent / 'weights' / 'track_predictor'
TRACK_PREDICTOR_MODEL: Path = find_weights("track_classifier")


log = logging.getLogger(__name__)

# Labels 0/1 → no debris, labels 2/3 → debris
BINARY_THRESHOLD: int = 2

def zone_from_path(path: Path) -> str:
    """Extract zone name from a gpkg or nc filename like ``Zone_Name_2025-02-04.gpkg``."""
    return re.sub(r'_\d{4}-\d{2}-\d{2}$', '', path.stem)

def _discover_run_files(
    runs_dir: Path,
    require_target: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Discover paired gpkg and nc files in a runs directory.

    Pairs each .nc file with the .gpkg that shares its zone prefix. Optionally
    filters to only nc files that contain ``unmasked_p_target``.

    Parameters
    ----------
    runs_dir : Path
        Directory containing .gpkg and .nc run file pairs.
    require_target : bool
        If True, skip nc files that lack ``unmasked_p_target``.

    Returns
    -------
    gpkg_paths : list[Path]
    nc_paths   : list[Path]
        Paired lists — gpkg_paths[i] corresponds to nc_paths[i].
    """
    zone_to_gpkg: dict[str, Path] = {
        zone_from_path(p): p for p in sorted(runs_dir.glob('*.gpkg'))
    }

    gpkg_paths: list[Path] = []
    nc_paths: list[Path] = []

    for nc_path in sorted(runs_dir.glob('*.nc')):
        zone = zone_from_path(nc_path)
        gpkg_path = zone_to_gpkg.get(zone)
        if gpkg_path is None:
            log.warning('discover_run_files: no gpkg for zone %s, skipping %s', zone, nc_path.name)
            continue

        if require_target:
            ds_peek = xr.open_dataset(nc_path)
            has_target = 'unmasked_p_target' in ds_peek.data_vars
            ds_peek.close()
            if not has_target:
                log.info('discover_run_files: %s lacks unmasked_p_target, skipping', nc_path.name)
                continue

        gpkg_paths.append(gpkg_path)
        nc_paths.append(nc_path)

    log.info('discover_run_files: found %d paired files in %s', len(nc_paths), runs_dir)
    return gpkg_paths, nc_paths


def build_training_set(
    labels: dict,
    runs_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and binary labels for all labeled tracks.

    Tracks with label >= BINARY_THRESHOLD (2 or 3) are treated as debris-present
    (y=1); labels 0 or 1 are debris-absent (y=0).

    Parameters
    ----------
    labels : dict
        Contents of ``track_labels.json``. Keys are ``zone|date|track_idx``;
        values have ``zone``, ``date``, ``track_idx``, ``label`` fields.
    runs_dir : Path
        Directory containing .gpkg and .nc run file pairs.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, one row per labeled track, indexed by label key.
    y : pd.Series
        Binary labels (0 = no debris, 1 = debris), same index as X.
    """
    # Discover which vars are needed: static + aspect + per-track orbit vars
    def _needed_vars(nc_path: Path) -> set[str]:
        ds_peek = xr.open_dataset(nc_path)
        orbit_vars = {v for v in ds_peek.data_vars
                      for pattern in _PER_TRACK_GROUPS.values() if pattern.match(v)}
        ds_peek.close()
        return set(STATIC_FEATURE_VARS) | {'slope', 'aspect'} | orbit_vars

    gpkg_paths, nc_paths = _discover_run_files(runs_dir)

    rows: list[dict] = []
    current_stem: str | None = None
    scene_ctx: dict | None = None

    for key, meta, row, ds in iter_labeled_run_tracks(gpkg_paths, nc_paths, labels):
        stem = f"{meta['zone']}_{meta['date']}"
        if stem != current_stem:
            scene_ctx = compute_scene_context(ds)
            current_stem = stem

        feats = extract_track_features(row, ds, scene_ctx=scene_ctx)
        feats['_key'] = key
        feats['_label'] = meta['label']
        rows.append(feats)

    if not rows:
        log.warning('build_training_set: no matching tracks found in %s', runs_dir)
        return pd.DataFrame(), pd.Series([], name='debris', dtype=int)

    df = pd.DataFrame(rows).set_index('_key')
    y = (df.pop('_label') >= BINARY_THRESHOLD).astype(int).rename('debris')

    log.info(
        'build_training_set: %d samples, %d features, %.0f%% positive',
        len(df), df.shape[1], 100 * y.mean(),
    )
    return df, y


def train_classifier(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Fit an XGBoost binary classifier on labeled tracks.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from build_training_set.
    y : pd.Series
        Binary labels (0 = no debris, 1 = debris).

    Returns
    -------
    XGBClassifier
        Fitted model. Feature names stored in clf.feature_names_in_.

    Notes
    -----
    Missing values are imputed with column medians before fitting.
    scale_pos_weight is set to n_neg/n_pos when negatives outnumber positives.
    """
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 and neg > pos else 1.0
    log.info(
        'train_classifier: %d pos / %d neg, scale_pos_weight=%.2f',
        pos, neg, scale_pos_weight,
    )

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
    )
    clf.fit(X.fillna(X.median()), y)

    log.debug(
        'train_classifier: train accuracy=%.3f',
        float((clf.predict(X.fillna(X.median())) == y).mean()),
    )
    return clf

def predict_track(clf, row, ds, scene_ctx):
    """Mostly a testing function for single tracks"""
    feats = extract_track_features(row, ds, scene_ctx=scene_ctx)
    X = pd.DataFrame([feats]).reindex(columns=clf.feature_names_in_).fillna(0)
    return float(clf.predict_proba(X)[0, 1])

def predict_tracks(
    clf: XGBClassifier,
    gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
) -> gpd.GeoDataFrame:
    """Score all tracks in a GeoDataFrame, adding a p_debris column.

    Parameters
    ----------
    clf : XGBClassifier
        Fitted classifier from train_classifier.
    gdf : gpd.GeoDataFrame
        Track polygons to score. Must be in the same CRS as ds.
    ds : xr.Dataset
        Dataset already reprojected to gdf.crs.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of gdf with an added p_debris column (float in [0, 1]).
    """
    scene_ctx = compute_scene_context(ds)
    feature_rows = [
        extract_track_features(row, ds, scene_ctx=scene_ctx)
        for _, row in gdf.iterrows()
    ]

    X_pred = pd.DataFrame(feature_rows, index=gdf.index)
    train_cols = list(clf.feature_names_in_)
    X_pred = X_pred.reindex(columns=train_cols).fillna(X_pred[train_cols].median())

    result = gdf.copy()
    result['p_debris'] = clf.predict_proba(X_pred)[:, 1]

    log.info(
        'predict_tracks: scored %d tracks, mean p_debris=%.3f',
        len(result), float(result['p_debris'].mean()),
    )
    return result