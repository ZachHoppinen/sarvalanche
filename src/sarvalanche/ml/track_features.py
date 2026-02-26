import itertools
import logging
import re

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask as _geom_mask
from rasterio.transform import from_bounds as _from_bounds
from scipy.ndimage import zoom as _zoom

log = logging.getLogger(__name__)

# ── Patch extraction constants ─────────────────────────────────────────────────
# Ordered channel list for extract_track_patch() output.
PATCH_CHANNELS: list[str] = [
    'distance_mahalanobis',  # 0 — primary detection signal
    'p_empirical',           # 1 — empirical detection probability
    'fcf',                   # 2 — forest cover fraction
    'slope',                 # 3 — terrain slope (radians)
    'northing',              # 4 — relative y position in patch, [-1 (bottom), +1 (top)]
    'easting',               # 5 — relative x position in patch, [-1 (left), +1 (right)]
    'track_mask',            # 6 — 1 inside track polygon, 0 outside
]
N_PATCH_CHANNELS: int = len(PATCH_CHANNELS)  # 7

_PATCH_DATA_VARS: list[str] = ['distance_mahalanobis', 'p_empirical', 'fcf', 'slope']
_TARGET_VAR: str = 'p_pixelwise'  # soft segmentation target for CNN training

# ── Static 2D terrain/physical variables — raw values, no probability transforms
STATIC_FEATURE_VARS: list[str] = [
    'fcf',         # forest cover fraction              (replaces p_fcf)
    'cell_counts', # FlowPy flow accumulation           (replaces p_runout)
    'slope',       # terrain slope angle                (replaces p_slope)
    'dem',         # elevation
]

# Per-track variable groups: pixel-wise max across all tracks of each type/pol.
# Keys become the feature prefix; patterns match scene-specific names like
# p_71_VV_empirical, d_93_VH_ml, etc.
_PER_TRACK_GROUPS: dict[str, re.Pattern] = {
    'empirical_VV': re.compile(r'^p_\d+_VV_empirical$'),
    'empirical_VH': re.compile(r'^p_\d+_VH_empirical$'),
    'd_ml_VV':      re.compile(r'^d_\d+_VV_ml$'),
    'd_ml_VH':      re.compile(r'^d_\d+_VH_ml$'),
}

_STATS: dict[str, callable] = {
    'mean': np.nanmean,
    'max':  np.nanmax,
    'std':  np.nanstd,
    'p75':  lambda v: np.nanpercentile(v, 75),
    'p90':  lambda v: np.nanpercentile(v, 90),
}


def _clip_arr(da: xr.DataArray, geom) -> np.ndarray:
    """Clip a DataArray to a geometry, flatten, and return as float (NaNs preserved)."""
    clipped = da.rio.clip([geom], all_touched=True, drop=True)
    return clipped.values.astype(float).ravel()


def _pixel_max_da(ds: xr.Dataset, var_list: list[str]) -> xr.DataArray | None:
    """
    Pixel-wise max across a list of same-shaped DataArrays.

    Returns None if ``var_list`` is empty. Inherits dims/coords and CRS from
    the first variable.
    """
    if not var_list:
        return None
    ref = ds[var_list[0]]
    stacked = np.stack([ds[v].values.astype(float) for v in var_list], axis=0)
    with np.errstate(all='ignore'):  # all-NaN slices at scene edges are expected
        max_arr = np.nanmax(stacked, axis=0)
    da = xr.DataArray(max_arr, dims=ref.dims, coords=ref.coords)
    da = da.rio.write_crs(ref.rio.crs)
    return da


def extract_track_features(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    static_vars: list[str] = STATIC_FEATURE_VARS,
) -> dict[str, float]:
    """
    Extract aggregate statistics and cross-variable correlations for a track polygon.

    For each static terrain variable and each per-track signal group the function
    computes mean/max/std/p75/p90 over the polygon pixels.  It also computes the
    Pearson correlation of pixel values between every (static_var, signal_group)
    pair, capturing spatial co-occurrence patterns such as whether the SAR change
    is concentrated on steep slopes or at high elevation.

    Per-track signal groups aggregate scene-specific orbit variables
    (e.g. ``p_71_VV_empirical``, ``d_93_VH_ml``) via pixel-wise max, producing
    stable feature names regardless of orbit IDs or track count.

    Parameters
    ----------
    row : gpd.GeoSeries
        A single track row. ``.geometry`` must be in the same CRS as ``ds``.
    ds : xr.Dataset
        Dataset already reprojected to match ``row.geometry``'s CRS. Must contain
        ``static_vars``; per-track vars are discovered automatically.
    static_vars : list[str]
        Terrain/physical variables to aggregate directly.

    Returns
    -------
    dict[str, float]
        ``{var}_{stat}`` for individual variables, ``{svar}_x_{group}_corr`` for
        cross-correlations, plus ``area_pixels``.

    Examples
    --------
    >>> feats = extract_track_features(gdf.loc[42], ds)
    >>> feats['slope_x_empirical_VH_corr']   # SAR change spatially correlated with slope?
    0.61
    """
    geom = row.geometry

    # ── Collect all clipped arrays once (NaN-preserving) ─────────────────────
    # Keys: static var names and per-track group names
    arrays: dict[str, np.ndarray] = {}

    for var in static_vars:
        if var not in ds:
            log.debug("extract_track_features: %s not in dataset, skipping", var)
            continue
        try:
            arrays[var] = _clip_arr(ds[var], geom)
        except Exception as exc:
            log.debug("extract_track_features: clip failed for %s – %s", var, exc)

    per_track_keys: list[str] = []
    for group, pattern in _PER_TRACK_GROUPS.items():
        matching = [v for v in ds.data_vars if pattern.match(v)]
        log.debug("extract_track_features: %s → %d vars", group, len(matching))
        da = _pixel_max_da(ds, matching)
        if da is None:
            continue
        try:
            arrays[group] = _clip_arr(da, geom)
            per_track_keys.append(group)
        except Exception as exc:
            log.debug("extract_track_features: clip failed for %s – %s", group, exc)

    # ── Per-variable aggregate statistics ─────────────────────────────────────
    features: dict[str, float] = {}
    area_pixels: int | None = None

    for name, arr in arrays.items():
        vals = arr[np.isfinite(arr)]
        if area_pixels is None:
            area_pixels = len(vals)
        if vals.size == 0:
            log.debug("extract_track_features: no valid pixels for %s", name)
            for stat in _STATS:
                features[f'{name}_{stat}'] = np.nan
            continue
        for stat, fn in _STATS.items():
            features[f'{name}_{stat}'] = float(fn(vals))

    def _corr(a: np.ndarray, b: np.ndarray, name: str) -> None:
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3:
            features[name] = np.nan
            return
        with np.errstate(invalid='ignore'):
            r = np.corrcoef(a[valid], b[valid])[0, 1]
        features[name] = float(r) if np.isfinite(r) else 0.0

    # ── Cross-correlations: each static terrain var × each per-track signal ───
    # Captures spatial co-occurrence: e.g. does the SAR signal occur where slope
    # is steep, or only at low elevation / low cell-count areas?
    for svar in static_vars:
        if svar not in arrays:
            continue
        for group in per_track_keys:
            _corr(arrays[svar], arrays[group], f'{svar}_x_{group}_corr')

    # ── Cross-correlations: per-track signal group pairs ──────────────────────
    # Captures signal agreement: if empirical change and ML distance both point
    # to the same pixels that is stronger evidence than either alone at medium
    # confidence.
    for ga, gb in itertools.combinations(per_track_keys, 2):
        _corr(arrays[ga], arrays[gb], f'{ga}_x_{gb}_corr')

    features['area_pixels'] = float(area_pixels) if area_pixels is not None else np.nan
    return features


def _patch_transform(ref_da: xr.DataArray, size: int):
    """Compute the affine transform mapping pixel coords to CRS coords for a patch.

    Returns the ``rasterio.transform.Affine`` for a ``(size, size)`` grid
    spanning the bounding box of ``ref_da`` (with half-pixel padding).
    """
    x_vals = ref_da.x.values
    y_vals = ref_da.y.values
    dx = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 30.0
    dy = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 30.0
    west  = float(x_vals.min()) - dx / 2
    east  = float(x_vals.max()) + dx / 2
    south = float(y_vals.min()) - dy / 2
    north = float(y_vals.max()) + dy / 2
    return _from_bounds(west, south, east, north, size, size)


def extract_track_patch(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    size: int = 64,
    buffer: float = 500.0,
) -> np.ndarray:
    """
    Extract a (C, size, size) float32 raster patch centred on a track polygon.

    Channel order matches ``PATCH_CHANNELS`` (indices 0-6):
    distance_mahalanobis, p_empirical, fcf, slope,
    northing (top=+1, bottom=-1), easting (left=-1, right=+1), track_mask.

    Northing and easting are always the canonical coordinate grid and are
    independent of raster values.  Missing data channels are filled with 0.

    Parameters
    ----------
    row : gpd.GeoSeries
        Single track row; ``.geometry`` must be in the same CRS as ``ds``.
    ds : xr.Dataset
        Dataset reprojected to match the track CRS.
    size : int
        Square output patch side length in pixels.
    buffer : float
        Context buffer in CRS units (metres for UTM) around the track bbox.

    Returns
    -------
    np.ndarray of shape (N_PATCH_CHANNELS, size, size)
    """
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds

    clip_sel = dict(
        x=slice(minx - buffer, maxx + buffer),
        y=slice(maxy + buffer, miny - buffer),  # y descending (raster convention)
    )

    # Find a reference variable to determine native clipped shape
    ref_var = next((v for v in _PATCH_DATA_VARS if v in ds.data_vars), None)
    if ref_var is None:
        log.warning("extract_track_patch: none of %s found in dataset", _PATCH_DATA_VARS)
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    ref_da = ds[ref_var].sel(**clip_sel)
    H, W = ref_da.shape

    if H < 2 or W < 2:
        log.debug("extract_track_patch: clipped region too small (%d×%d), returning zeros", H, W)
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    out = np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)
    zoom_y, zoom_x = size / H, size / W

    # Channels 0-3: raster data variables
    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var not in ds.data_vars:
            continue
        arr = ds[var].sel(**clip_sel).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        out[ch] = _zoom(arr, (zoom_y, zoom_x), order=1) if arr.shape != (size, size) else arr

    # Channel 4: northing — linear top=+1 → bottom=-1 (constant across all samples)
    north_vec = np.linspace(1.0, -1.0, size, dtype=np.float32)
    out[4] = np.broadcast_to(north_vec[:, np.newaxis], (size, size)).copy()

    # Channel 5: easting — linear left=-1 → right=+1
    east_vec = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    out[5] = np.broadcast_to(east_vec[np.newaxis, :], (size, size)).copy()

    # Channel 6: polygon mask at target resolution
    transform = _patch_transform(ref_da, size)
    out[6] = (~_geom_mask([geom], out_shape=(size, size), transform=transform,
                           all_touched=True)).astype(np.float32)

    return out


def extract_track_patch_with_target(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    size: int = 64,
    buffer: float = 500.0,
    debris_shapes: gpd.GeoDataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch and its ``p_pixelwise`` soft segmentation target.

    Returns the same ``(C, size, size)`` patch as ``extract_track_patch`` plus a
    ``(1, size, size)`` target map from ``p_pixelwise``, used for training the
    segmentation head. If ``p_pixelwise`` is not in the dataset the target is
    all zeros.

    When ``debris_shapes`` is provided (a GeoDataFrame of manually drawn debris
    polygons), they are rasterized onto the patch grid and blended with the
    ``p_pixelwise`` target via element-wise max, reinforcing the training signal
    with manual annotations.

    Parameters
    ----------
    row, ds, size, buffer
        Same as ``extract_track_patch``.
    debris_shapes : gpd.GeoDataFrame or None
        Optional drawn debris polygons for this track. Must be in the same CRS
        as ``ds``. Geometries that don't intersect the patch are harmless.

    Returns
    -------
    patch : np.ndarray of shape (N_PATCH_CHANNELS, size, size)
    target : np.ndarray of shape (1, size, size)
    """
    patch = extract_track_patch(row, ds, size=size, buffer=buffer)

    target = np.zeros((1, size, size), dtype=np.float32)
    if _TARGET_VAR not in ds.data_vars:
        log.debug("extract_track_patch_with_target: %s not in dataset", _TARGET_VAR)
        return patch, target

    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    clip_sel = dict(
        x=slice(minx - buffer, maxx + buffer),
        y=slice(maxy + buffer, miny - buffer),
    )

    ref_var = next((v for v in _PATCH_DATA_VARS if v in ds.data_vars), None)
    ref_da = ds[ref_var].sel(**clip_sel) if ref_var else None

    arr = ds[_TARGET_VAR].sel(**clip_sel).values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    H, W = arr.shape
    if H >= 2 and W >= 2:
        zoom_y, zoom_x = size / H, size / W
        target[0] = _zoom(arr, (zoom_y, zoom_x), order=1) if arr.shape != (size, size) else arr

    # Blend manually drawn debris shapes via element-wise max
    if debris_shapes is not None and not debris_shapes.empty and ref_da is not None:
        rH, rW = ref_da.shape
        if rH >= 2 and rW >= 2:
            transform = _patch_transform(ref_da, size)
            geoms = list(debris_shapes.geometry)
            shape_mask = (~_geom_mask(geoms, out_shape=(size, size),
                                      transform=transform,
                                      all_touched=True)).astype(np.float32)
            target[0] = np.maximum(target[0], shape_mask)

    return patch, target


# ── Segmentation feature aggregation ─────────────────────────────────────────

_SEG_FEATURE_NAMES: list[str] = [
    'seg_mean', 'seg_max', 'seg_p75', 'seg_p90', 'seg_p95', 'seg_frac_above_05',
]


def aggregate_seg_features(
    seg_map: np.ndarray,
    track_mask: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate a segmentation map within the track polygon mask into scalar features.

    Parameters
    ----------
    seg_map : np.ndarray of shape (H, W)
        Sigmoid-activated segmentation probabilities.
    track_mask : np.ndarray of shape (H, W)
        Binary mask (1 inside track polygon, 0 outside).

    Returns
    -------
    dict with keys: seg_mean, seg_max, seg_p75, seg_p90, seg_p95, seg_frac_above_05
    """
    vals = seg_map[track_mask > 0.5]
    if vals.size == 0:
        return {k: np.nan for k in _SEG_FEATURE_NAMES}

    return {
        'seg_mean': float(np.nanmean(vals)),
        'seg_max': float(np.nanmax(vals)),
        'seg_p75': float(np.nanpercentile(vals, 75)),
        'seg_p90': float(np.nanpercentile(vals, 90)),
        'seg_p95': float(np.nanpercentile(vals, 95)),
        'seg_frac_above_05': float((vals > 0.5).sum() / len(vals)),
    }
