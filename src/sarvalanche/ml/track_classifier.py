import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from xgboost import XGBClassifier

from sarvalanche.ml.track_features import (
    STATIC_FEATURE_VARS,
    TRACK_MASK_CHANNEL,
    aggregate_seg_features,
    extract_track_features,
    extract_track_patch,
    extract_track_patch_with_target,
)

log = logging.getLogger(__name__)

from sarvalanche.ml.weight_utils import find_weights

TRACK_PREDICTOR_DIR: Path = Path(__file__).parent / 'weights' / 'track_predictor'
TRACK_PREDICTOR_MODEL: Path = find_weights("track_classifier")

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

        for key, meta in tqdm(entries, desc=stem, unit="trk"):
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
    seg_encoder=None,
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
    seg_encoder : TrackSegEncoder, optional
        If provided, runs segmentation and appends aggregated seg features.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of ``gdf`` with an added ``p_debris`` column (float in [0, 1]).
    """
    import torch

    feature_rows = []
    for _, row in gdf.iterrows():
        feats = extract_track_features(row, ds)

        if seg_encoder is not None:
            patch = extract_track_patch(row, ds)
            with torch.no_grad():
                seg_logits = seg_encoder.segment(torch.FloatTensor(patch[np.newaxis]))
                seg_probs = torch.sigmoid(seg_logits).numpy()[0, 0]  # (H, W)
            track_mask = patch[TRACK_MASK_CHANNEL]
            feats.update(aggregate_seg_features(seg_probs, track_mask))

        feature_rows.append(feats)

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


def build_patch_training_set(
    labels: dict,
    runs_dir: Path,
    size: int = 64,
) -> tuple[np.ndarray, pd.Series]:
    """
    Extract multi-channel raster patches and binary labels for all labeled tracks.

    Mirrors ``build_training_set`` but returns spatial patch arrays instead of
    aggregate feature vectors.  Files are loaded once per stem to minimise I/O.

    Parameters
    ----------
    labels : dict
        Contents of ``track_labels.json``.
    runs_dir : Path
        Directory containing ``.gpkg`` and ``.nc`` run file pairs.
    size : int
        Output patch side length in pixels (passed to ``extract_track_patch``).

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
        Float32 patch array; channel order matches ``PATCH_CHANNELS``.
    y : pd.Series
        Binary labels indexed by label key (0 = no debris, 1 = debris).
    """
    from sarvalanche.ml.track_features import N_PATCH_CHANNELS

    by_file: dict[str, list] = {}
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_file.setdefault(stem, []).append((key, meta))

    patch_list:  list[np.ndarray] = []
    key_list:    list[str]        = []
    label_list:  list[int]        = []

    for stem, entries in by_file.items():
        gpkg_path = runs_dir / f"{stem}.gpkg"
        nc_path   = runs_dir / f"{stem}.nc"
        if not gpkg_path.exists() or not nc_path.exists():
            log.warning("build_patch_training_set: missing files for %s, skipping", stem)
            continue

        log.info("build_patch_training_set: extracting patches from %s (%d tracks)",
                 stem, len(entries))
        gdf = gpd.read_file(gpkg_path)
        ds  = _load_ds(nc_path, gdf.crs)

        for key, meta in tqdm(entries, desc=stem, unit="trk"):
            idx = meta['track_idx']
            if idx not in gdf.index:
                log.warning("build_patch_training_set: track %d not in %s, skipping", idx, stem)
                continue
            patch = extract_track_patch(gdf.loc[idx], ds, size=size)
            patch_list.append(patch)
            key_list.append(key)
            label_list.append(meta['label'])

        ds.close()

    if not patch_list:
        return np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32), \
               pd.Series([], name='debris', dtype=int)

    patches = np.stack(patch_list, axis=0)
    y = pd.Series(
        [int(lbl >= BINARY_THRESHOLD) for lbl in label_list],
        index=key_list,
        name='debris',
        dtype=int,
    )
    log.info(
        "build_patch_training_set: %d patches, %.0f%% positive",
        len(patches), 100 * y.mean(),
    )
    return patches, y


def build_seg_training_set(
    labels: dict,
    runs_dir: Path,
    size: int = 64,
    shapes_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Extract patches, pixel-wise soft targets, and binary labels for segmentation training.

    Like ``build_patch_training_set`` but also extracts ``unmasked_p_target`` as a
    ``(1, size, size)`` target for each track.

    When ``shapes_path`` points to a GeoPackage of manually drawn debris
    polygons (with a ``key`` column matching label keys), the shapes are
    rasterized onto the patch grid and blended with ``unmasked_p_target`` via
    element-wise max.

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
    targets : np.ndarray of shape (N, 1, size, size)
    y : pd.Series  — binary labels indexed by label key
    """
    from sarvalanche.ml.track_features import N_PATCH_CHANNELS

    # Load drawn debris shapes once, grouped by label key
    shapes_by_key: dict[str, gpd.GeoDataFrame] = {}
    if shapes_path is not None and shapes_path.exists():
        all_shapes = gpd.read_file(shapes_path)
        if not all_shapes.empty and 'key' in all_shapes.columns:
            for k, grp in all_shapes.groupby('key'):
                shapes_by_key[k] = grp
            log.info("build_seg_training_set: loaded %d debris shapes (%d keys)",
                     len(all_shapes), len(shapes_by_key))

    by_file: dict[str, list] = {}
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_file.setdefault(stem, []).append((key, meta))

    patch_list:  list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    key_list:    list[str]        = []
    label_list:  list[int]        = []

    for stem, entries in by_file.items():
        gpkg_path = runs_dir / f"{stem}.gpkg"
        nc_path   = runs_dir / f"{stem}.nc"
        if not gpkg_path.exists() or not nc_path.exists():
            log.warning("build_seg_training_set: missing files for %s, skipping", stem)
            continue

        log.info("build_seg_training_set: extracting patches from %s (%d tracks)",
                 stem, len(entries))
        gdf = gpd.read_file(gpkg_path)
        ds  = _load_ds(nc_path, gdf.crs)

        for key, meta in tqdm(entries, desc=stem, unit="trk"):
            idx = meta['track_idx']
            if idx not in gdf.index:
                log.warning("build_seg_training_set: track %d not in %s, skipping", idx, stem)
                continue
            # Get drawn shapes for this track (reprojected to dataset CRS)
            track_shapes = shapes_by_key.get(key)
            if track_shapes is not None and track_shapes.crs is not None and track_shapes.crs != gdf.crs:
                track_shapes = track_shapes.to_crs(gdf.crs)
            patch, target = extract_track_patch_with_target(
                gdf.loc[idx], ds, size=size, debris_shapes=track_shapes,
            )
            # For no-debris tracks, zero out the target inside the track polygon
            # so the model learns to suppress false positives there.
            if meta['label'] < BINARY_THRESHOLD:
                track_mask = patch[TRACK_MASK_CHANNEL]
                target[0] *= (1.0 - track_mask)
            patch_list.append(patch)
            target_list.append(target)
            key_list.append(key)
            label_list.append(meta['label'])

        ds.close()

    if not patch_list:
        return (
            np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            np.empty((0, 1, size, size), dtype=np.float32),
            pd.Series([], name='debris', dtype=int),
        )

    patches = np.stack(patch_list, axis=0)
    targets = np.stack(target_list, axis=0)
    y = pd.Series(
        [int(lbl >= BINARY_THRESHOLD) for lbl in label_list],
        index=key_list,
        name='debris',
        dtype=int,
    )
    log.info(
        "build_seg_training_set: %d patches, %.0f%% positive",
        len(patches), 100 * y.mean(),
    )
    return patches, targets, y


def _zone_from_gpkg(gpkg_path: Path) -> str:
    """Extract zone name from a gpkg filename like ``Zone_Name_2025-02-04.gpkg``."""
    import re
    return re.sub(r'_\d{4}-\d{2}-\d{2}$', '', gpkg_path.stem)


def build_all_seg_patches(
    runs_dir: Path,
    size: int = 64,
    max_tracks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patches and pixel-wise soft targets for *all* tracks across all dates.

    Each ``.gpkg`` defines track polygons for a zone.  Every ``.nc`` file that
    shares the same zone prefix **and** contains ``unmasked_p_target`` is paired with
    that gpkg, producing one (patch, target) per (track, date) combination.

    Unlike ``build_seg_training_set``, this does not require labels.
    ``unmasked_p_target`` is used as-is (no negative-label zeroing, no debris-shape
    blending).

    Parameters
    ----------
    runs_dir : Path
        Directory containing ``.gpkg`` and ``.nc`` run file pairs.
    size : int
        Output patch side length in pixels.
    max_tracks : int or None
        If set, stop after extracting this many patches (for quick testing).

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
    targets : np.ndarray of shape (N, 1, size, size)
    """
    from tqdm import tqdm

    from sarvalanche.ml.track_features import N_PATCH_CHANNELS

    gpkg_paths = sorted(runs_dir.glob('*.gpkg'))
    if not gpkg_paths:
        log.warning("build_all_seg_patches: no .gpkg files in %s", runs_dir)
        return (
            np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            np.empty((0, 1, size, size), dtype=np.float32),
        )

    # Group nc files by zone prefix
    all_nc = sorted(runs_dir.glob('*.nc'))
    nc_by_zone: dict[str, list[Path]] = {}
    for nc in all_nc:
        zone = _zone_from_gpkg(nc)
        nc_by_zone.setdefault(zone, []).append(nc)

    patch_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    for gpkg_path in gpkg_paths:
        zone = _zone_from_gpkg(gpkg_path)
        nc_paths = nc_by_zone.get(zone, [])
        if not nc_paths:
            log.warning("build_all_seg_patches: no .nc files for zone %s, skipping", zone)
            continue

        gdf = gpd.read_file(gpkg_path)

        for nc_path in nc_paths:
            # Skip nc files that lack unmasked_p_target (e.g. early incomplete runs)
            ds_peek = xr.open_dataset(nc_path)
            has_target = 'unmasked_p_target' in ds_peek.data_vars
            ds_peek.close()
            if not has_target:
                log.info("build_all_seg_patches: %s lacks unmasked_p_target, skipping",
                         nc_path.name)
                continue

            ds = _load_ds(nc_path, gdf.crs)
            desc = f"{zone} | {nc_path.stem.split('_')[-1]}"
            log.info("build_all_seg_patches: %d tracks x %s", len(gdf), nc_path.name)

            for idx in tqdm(gdf.index, desc=desc, unit="trk"):
                try:
                    patch, target = extract_track_patch_with_target(
                        gdf.loc[idx], ds, size=size,
                    )
                except Exception:
                    log.debug("build_all_seg_patches: failed on track %d in %s",
                              idx, nc_path.name, exc_info=True)
                    continue
                patch_list.append(patch)
                target_list.append(target)
                if max_tracks is not None and len(patch_list) >= max_tracks:
                    break

            ds.close()
            if max_tracks is not None and len(patch_list) >= max_tracks:
                log.info("build_all_seg_patches: hit max_tracks=%d, stopping early", max_tracks)
                break
        if max_tracks is not None and len(patch_list) >= max_tracks:
            break

    if not patch_list:
        return (
            np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            np.empty((0, 1, size, size), dtype=np.float32),
        )

    patches = np.stack(patch_list, axis=0)
    targets = np.stack(target_list, axis=0)
    log.info("build_all_seg_patches: %d total patches (%d zones, %d nc files)",
             len(patches), len(gpkg_paths), sum(1 for ncs in nc_by_zone.values() for _ in ncs))
    return patches, targets
