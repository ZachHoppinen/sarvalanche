"""
Training data assembly for the debris segmentation CNN.

Three public functions:

  - build_seg_training_set   : labeled tracks only, with manual debris polygons
                               blended into targets. Returns in-memory arrays.
  - build_all_seg_patches    : all tracks across all dates using unmasked_p_target
                               as weak supervision. Writes to disk-backed memmap.
  - build_patch_training_set : context patches + binary labels for track-level
                               CNN classification (not segmentation).

All functions use load_netcdf_to_dataset (no reprojection) and pass src_crs
to patch extractors so geometry reprojection happens at clip time.
"""

import gc
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.track_classifier import _discover_run_files, zone_from_path
from sarvalanche.ml.track_classifier import BINARY_THRESHOLD
from sarvalanche.ml.track_patch_extraction import (
    N_PATCH_CHANNELS,
    extract_context_patch,
    extract_dual_scale_with_target,
)

log = logging.getLogger(__name__)


def build_seg_training_set(
    labels: dict,
    runs_dir: Path,
    size: int = 64,
    shapes_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Extract context patches, pixel-wise soft targets, and binary labels.

    Parameters
    ----------
    labels : dict
        Contents of ``track_labels.json``.
    runs_dir : Path
        Directory containing .gpkg and .nc run file pairs.
    size : int
        Output patch size in pixels.
    shapes_path : Path or None
        Optional GeoPackage of manually drawn debris polygons with a ``key``
        column matching label keys.

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
    targets : np.ndarray of shape (N, 1, size, size)
    y : pd.Series — binary labels indexed by label key
    """
    # Load manual debris shapes grouped by label key
    shapes_by_key: dict[str, gpd.GeoDataFrame] = {}
    if shapes_path is not None and shapes_path.exists():
        all_shapes = gpd.read_file(shapes_path)
        if not all_shapes.empty and 'key' in all_shapes.columns:
            for k, grp in all_shapes.groupby('key'):
                shapes_by_key[k] = grp
            log.info(
                'build_seg_training_set: loaded %d debris shapes (%d keys)',
                len(all_shapes), len(shapes_by_key),
            )

    # Group labels by stem for file-efficient iteration
    from collections import defaultdict
    by_stem: dict[str, list] = defaultdict(list)
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_stem[stem].append((key, meta))

    gpkg_paths, nc_paths = _discover_run_files(runs_dir)
    zone_to_gpkg = {zone_from_path(p): p for p in gpkg_paths}

    patch_list:  list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    key_list:    list[str]        = []
    label_list:  list[int]        = []

    for nc_path in nc_paths:
        stem = nc_path.stem
        entries = by_stem.get(stem)
        if not entries:
            continue

        zone = zone_from_path(nc_path)
        gpkg_path = zone_to_gpkg.get(zone)
        if gpkg_path is None:
            continue

        gdf = gpd.read_file(gpkg_path)
        ds = load_netcdf_to_dataset(nc_path)
        log.info(
            'build_seg_training_set: %d labeled tracks — %s',
            len(entries), nc_path.name,
        )

        for key, meta in tqdm(entries, desc=stem, unit='trk'):
            # if no debris labeled skip.
            if meta['label'] < BINARY_THRESHOLD:
                continue

            idx = meta['track_idx']
            if idx not in gdf.index:
                log.warning(
                    'build_seg_training_set: track %d not in %s, skipping',
                    idx, stem,
                )
                continue

            track_shapes = shapes_by_key.get(key)
            if track_shapes is not None and track_shapes.crs != gdf.crs:
                track_shapes = track_shapes.to_crs(gdf.crs)

            try:
                dual, targets = extract_dual_scale_with_target(
                    gdf.loc[idx], ds, size=size,
                    src_crs=gdf.crs,
                    debris_shapes=track_shapes,
                )
                patch = dual.context
                target = targets[0]  # context target only
            except Exception as exc:
                log.warning(
                    'build_seg_training_set: failed on track %d in %s — %s',
                    idx, stem, exc,
                )
                continue

            # Zero out target inside track polygon for no-debris tracks
            # if meta['label'] < BINARY_THRESHOLD:
                # from sarvalanche.ml.track_patch_extraction import TRACK_MASK_CHANNEL
                # track_mask = patch[TRACK_MASK_CHANNEL]
                # target[0] *= (1.0 - track_mask)

            patch_list.append(patch)
            target_list.append(target)
            key_list.append(key)
            label_list.append(meta['label'])

        ds.close()
        gc.collect()

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
        'build_seg_training_set: %d patches, %.0f%% positive',
        len(patches), 100 * y.mean(),
    )
    return patches, targets, y


def build_all_seg_patches(
    runs_dir: Path,
    size: int = 64,
    max_tracks: int | None = None,
    memmap_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract context patches and weak targets for all tracks across all dates.

    Uses unmasked_p_target as-is for all tracks — no label filtering, no
    debris-shape blending. Suitable for pre-training or self-supervised warmup.

    Parameters
    ----------
    runs_dir : Path
    size : int
        Output patch size in pixels.
    max_tracks : int or None
        Stop after this many patches (for testing).
    memmap_dir : Path or None
        If provided, write patches to disk-backed binary files and return
        memory-mapped views. If None, accumulate in RAM.

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
    targets : np.ndarray of shape (N, 1, size, size)
    """
    from sarvalanche.ml.track_patch_extraction import _TARGET_VAR

    gpkg_paths, nc_paths = _discover_run_files(runs_dir, require_target=True)
    zone_to_gpkg = {zone_from_path(p): p for p in gpkg_paths}

    use_disk = memmap_dir is not None
    if use_disk:
        memmap_dir = Path(memmap_dir)
        memmap_dir.mkdir(parents=True, exist_ok=True)
        patches_fp = memmap_dir / 'patches.bin'
        targets_fp = memmap_dir / 'targets.bin'
        patch_fh = None
        target_fh = None
        write_count = 0
    else:
        patch_list:  list[np.ndarray] = []
        target_list: list[np.ndarray] = []

    if use_disk:
        patch_fh = open(patches_fp, 'wb')
        target_fh = open(targets_fp, 'wb')

    try:
        done = False
        for nc_path in nc_paths:
            if done:
                break
            zone = zone_from_path(nc_path)
            gpkg_path = zone_to_gpkg.get(zone)
            if gpkg_path is None:
                continue

            gdf = gpd.read_file(gpkg_path)
            ds = load_netcdf_to_dataset(nc_path)
            log.info(
                'build_all_seg_patches: %d tracks — %s', len(gdf), nc_path.name
            )

            for idx in tqdm(gdf.index, desc=nc_path.stem, unit='trk'):
                try:
                    dual, targets = extract_dual_scale_with_target(
                        gdf.loc[idx], ds, size=size, src_crs=gdf.crs,
                    )
                    patch  = dual.context
                    target = targets[0]
                except Exception:
                    log.debug(
                        'build_all_seg_patches: failed on track %d in %s',
                        idx, nc_path.name, exc_info=True,
                    )
                    continue

                expected_p = (N_PATCH_CHANNELS, size, size)
                expected_t = (1, size, size)
                if patch.shape != expected_p or target.shape != expected_t:
                    log.debug(
                        'build_all_seg_patches: skipping track %d — shape mismatch', idx
                    )
                    continue

                if use_disk:
                    patch_fh.write(patch.astype(np.float32).tobytes())
                    target_fh.write(target.astype(np.float32).tobytes())
                    write_count += 1
                else:
                    patch_list.append(patch)
                    target_list.append(target)

                count = write_count if use_disk else len(patch_list)
                if max_tracks is not None and count >= max_tracks:
                    done = True
                    log.info(
                        'build_all_seg_patches: hit max_tracks=%d, stopping', max_tracks
                    )
                    break

            ds.close()
            gc.collect()
    finally:
        if use_disk:
            patch_fh.close()
            target_fh.close()

        if write_count == 0:
            patches_fp.unlink(missing_ok=True)
            targets_fp.unlink(missing_ok=True)
            return (
                np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
                np.empty((0, 1, size, size), dtype=np.float32),
            )

        patch_bytes  = N_PATCH_CHANNELS * size * size * 4
        target_bytes = 1 * size * size * 4
        actual_p = patches_fp.stat().st_size // patch_bytes
        actual_t = targets_fp.stat().st_size // target_bytes
        if actual_p != write_count or actual_t != write_count:
            log.warning(
                'build_all_seg_patches: size mismatch — write_count=%d, '
                'patch_file=%d, target_file=%d. Using min.',
                write_count, actual_p, actual_t,
            )
            write_count = min(actual_p, actual_t)

        patches_out = np.memmap(
            patches_fp, dtype=np.float32, mode='r',
            shape=(write_count, N_PATCH_CHANNELS, size, size),
        )
        targets_out = np.memmap(
            targets_fp, dtype=np.float32, mode='r',
            shape=(write_count, 1, size, size),
        )
        log.info(
            'build_all_seg_patches: %d patches on disk (%.1f GB)',
            write_count,
            (patches_fp.stat().st_size + targets_fp.stat().st_size) / 1e9,
        )
        return patches_out, targets_out

    if not patch_list:
        return (
            np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            np.empty((0, 1, size, size), dtype=np.float32),
        )

    return np.stack(patch_list), np.stack(target_list)


def build_patch_training_set(
    labels: dict,
    runs_dir: Path,
    size: int = 64,
) -> tuple[np.ndarray, pd.Series]:
    """Extract context patches and binary labels for track-level classification.

    Parameters
    ----------
    labels : dict
    runs_dir : Path
    size : int

    Returns
    -------
    patches : np.ndarray of shape (N, C, size, size)
    y : pd.Series — binary labels indexed by label key
    """
    from collections import defaultdict
    by_stem: dict[str, list] = defaultdict(list)
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_stem[stem].append((key, meta))

    gpkg_paths, nc_paths = _discover_run_files(runs_dir)
    zone_to_gpkg = {zone_from_path(p): p for p in gpkg_paths}

    patch_list: list[np.ndarray] = []
    key_list:   list[str]        = []
    label_list: list[int]        = []

    for nc_path in nc_paths:
        stem = nc_path.stem
        entries = by_stem.get(stem)
        if not entries:
            continue

        zone = zone_from_path(nc_path)
        gpkg_path = zone_to_gpkg.get(zone)
        if gpkg_path is None:
            continue

        gdf = gpd.read_file(gpkg_path)
        ds = load_netcdf_to_dataset(nc_path)

        for key, meta in tqdm(entries, desc=stem, unit='trk'):
            idx = meta['track_idx']
            if idx not in gdf.index:
                continue
            try:
                patch = extract_context_patch(
                    gdf.loc[idx], ds, size=size, src_crs=gdf.crs,
                )
            except Exception as exc:
                log.warning(
                    'build_patch_training_set: failed on track %d — %s', idx, exc
                )
                continue
            patch_list.append(patch)
            key_list.append(key)
            label_list.append(meta['label'])

        ds.close()
        gc.collect()

    if not patch_list:
        return (
            np.empty((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            pd.Series([], name='debris', dtype=int),
        )

    y = pd.Series(
        [int(lbl >= BINARY_THRESHOLD) for lbl in label_list],
        index=key_list,
        name='debris',
        dtype=int,
    )
    log.info(
        'build_patch_training_set: %d patches, %.0f%% positive',
        len(patch_list), 100 * y.mean(),
    )
    return np.stack(patch_list), y