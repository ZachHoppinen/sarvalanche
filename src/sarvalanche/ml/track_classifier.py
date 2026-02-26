import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier

from sarvalanche.ml.track_features import STATIC_FEATURE_VARS, extract_track_features

log = logging.getLogger(__name__)

# Canonical location for saved track predictor artefacts.
# Resolves to <project_root>/ml/weights/track_predictor/ regardless of where
# the module is imported from.
TRACK_PREDICTOR_DIR: Path = Path(__file__).parents[3] / 'ml' / 'weights' / 'track_predictor'
TRACK_PREDICTOR_MODEL: Path = TRACK_PREDICTOR_DIR / 'track_classifier.joblib'

# Labels 0/1 → no debris, labels 2/3 → debris
BINARY_THRESHOLD: int = 2


def _load_ds(nc_path: Path, target_crs) -> xr.Dataset:
    """Load all 2D (y, x) variables from a run NetCDF, reprojecting to target_crs.

    Loading all 2D vars ensures that scene-specific per-track variables
    (e.g. ``p_71_VV_empirical``, ``d_93_VH_ml``) are available for dynamic
    discovery in ``extract_track_features``, regardless of orbit IDs.

    The run NetCDFs are stored in EPSG:4326; CRS metadata is written before
    reprojecting so rioxarray can perform the transformation correctly.

    Parameters
    ----------
    nc_path : Path
        Path to a ``*_YYYY-MM-DD.nc`` run file.
    target_crs :
        Any CRS accepted by rioxarray (e.g. ``gdf.crs``). Typically EPSG:32611.

    Returns
    -------
    xr.Dataset
        Dataset in ``target_crs`` containing all 2D variables.
    """
    ds_peek = xr.open_dataset(nc_path)
    load_vars = [v for v in ds_peek.data_vars if ds_peek[v].dims == ('y', 'x')]
    ds_peek.close()

    ds = (
        xr.open_dataset(nc_path)[load_vars]
        .astype(float)
        .rio.write_crs('EPSG:4326')
        .rio.reproject(target_crs)
    )
    log.debug("_load_ds: loaded %s — %d 2D vars", nc_path.name, len(load_vars))
    return ds


def build_training_set(
    labels: dict,
    runs_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and binary labels for all labeled tracks.

    Tracks with label >= ``BINARY_THRESHOLD`` (2 or 3) are treated as
    debris-present (y=1); labels 0 or 1 are debris-absent (y=0).

    Parameters
    ----------
    labels : dict
        Contents of ``track_labels.json``. Keys are ``zone|date|track_idx``;
        values have ``zone``, ``date``, ``track_idx``, ``label`` fields.
    runs_dir : Path
        Directory containing ``.gpkg`` and ``.nc`` run file pairs.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, one row per labeled track, indexed by label key.
    y : pd.Series
        Binary labels (0 = no debris, 1 = debris), same index as X.

    Examples
    --------
    >>> X, y = build_training_set(labels, runs_dir)
    >>> X.shape
    (55, 41)
    >>> y.value_counts()
    debris
    1    38
    0    17
    """
    # Group by file to minimise expensive reproject calls
    by_file: dict[str, list] = {}
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_file.setdefault(stem, []).append((key, meta))

    rows: list[dict] = []
    for stem, entries in by_file.items():
        gpkg_path = runs_dir / f"{stem}.gpkg"
        nc_path = runs_dir / f"{stem}.nc"
        if not gpkg_path.exists() or not nc_path.exists():
            log.warning("build_training_set: missing files for %s, skipping", stem)
            continue

        log.info("build_training_set: extracting features from %s (%d tracks)",
                 stem, len(entries))
        gdf = gpd.read_file(gpkg_path)
        ds = _load_ds(nc_path, gdf.crs)

        for key, meta in entries:
            idx = meta['track_idx']
            if idx not in gdf.index:
                log.warning("build_training_set: track %d not in %s, skipping", idx, stem)
                continue
            feats = extract_track_features(gdf.loc[idx], ds)
            feats['_key'] = key
            feats['_label'] = meta['label']
            rows.append(feats)

        ds.close()

    df = pd.DataFrame(rows).set_index('_key')
    y = (df.pop('_label') >= BINARY_THRESHOLD).astype(int).rename('debris')
    X = df

    log.info(
        "build_training_set: %d samples, %d features, %.0f%% positive",
        len(X), X.shape[1], 100 * y.mean(),
    )
    return X, y


def train_classifier(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """
    Fit an XGBoost binary classifier on labeled tracks.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from ``build_training_set``.
    y : pd.Series
        Binary labels (0 = no debris, 1 = debris).

    Returns
    -------
    XGBClassifier
        Fitted model. Feature names are stored in ``clf.feature_names_in_``.

    Notes
    -----
    Missing feature values are imputed with column medians before fitting.
    ``scale_pos_weight`` is set to ``n_neg / n_pos`` only when negative
    samples outnumber positives, otherwise left at 1.0.
    """
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 and neg > pos else 1.0
    log.info(
        "train_classifier: %d pos / %d neg, scale_pos_weight=%.2f",
        pos, neg, scale_pos_weight,
    )

    X_filled = X.fillna(X.median())

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
    )
    clf.fit(X_filled, y)

    log.debug(
        "train_classifier: train accuracy=%.3f",
        float((clf.predict(X_filled) == y).mean()),
    )
    return clf


def predict_tracks(
    clf: XGBClassifier,
    gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
) -> gpd.GeoDataFrame:
    """
    Score all tracks in a GeoDataFrame, adding a ``p_debris`` column.

    Parameters
    ----------
    clf : XGBClassifier
        Fitted classifier from ``train_classifier``.
    gdf : gpd.GeoDataFrame
        Track polygons to score. Must be in the same CRS as ``ds``.
    ds : xr.Dataset
        Dataset already reprojected to ``gdf.crs``.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of ``gdf`` with an added ``p_debris`` column (float in [0, 1]).

    Examples
    --------
    >>> gdf_scored = predict_tracks(clf, gdf, ds)
    >>> gdf_scored['p_debris'].describe()
    """
    feature_rows = [extract_track_features(row, ds) for _, row in gdf.iterrows()]
    X_pred = pd.DataFrame(feature_rows, index=gdf.index)

    # Align to training features; fill unseen columns with median, drop extras
    train_cols = list(clf.feature_names_in_)
    medians = X_pred[train_cols].median()
    X_pred = X_pred.reindex(columns=train_cols).fillna(medians)

    proba = clf.predict_proba(X_pred)[:, 1]
    result = gdf.copy()
    result['p_debris'] = proba

    log.info(
        "predict_tracks: scored %d tracks, mean p_debris=%.3f",
        len(result), float(proba.mean()),
    )
    return result
