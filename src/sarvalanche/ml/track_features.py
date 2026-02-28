import itertools
import logging
import re

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask as _geom_mask
from rasterio.transform import from_bounds as _from_bounds
from scipy.ndimage import label as _label
from scipy.ndimage import zoom as _zoom

log = logging.getLogger(__name__)

# ── Patch extraction constants ─────────────────────────────────────────────────
# Ordered channel list for extract_track_patch() output.
PATCH_CHANNELS: list[str] = [
    'combined_distance',     # 0 — ML z-score distance (signed, std devs)
    'd_empirical',           # 1 — combined backscatter change (dB)
    'fcf',                   # 2 — forest cover fraction
    'slope',                 # 3 — terrain slope (radians)
    'cell_counts',           # 4 — FlowPy runout cell counts
    'northing',              # 5 — relative y position in patch, [-1 (bottom), +1 (top)]
    'easting',               # 6 — relative x position in patch, [-1 (left), +1 (right)]
    'track_mask',            # 7 — 1 inside track polygon, 0 outside
]
N_PATCH_CHANNELS: int = len(PATCH_CHANNELS)  # 8

_PATCH_DATA_VARS: list[str] = [
    'combined_distance', 'd_empirical', 'fcf', 'slope', 'cell_counts',
]
_TARGET_VAR: str = 'p_pixelwise'  # soft segmentation target for CNN training

# ── Per-channel normalization ─────────────────────────────────────────────────
# Fixed constants applied in extract_track_patch() so all channels are roughly
# in the same scale when entering the CNN.  'log1p' means apply np.log1p(|x|)
# with sign preservation before dividing.
#
# Channels not listed here (northing, easting, track_mask) are already in
# [-1, 1] or {0, 1} and need no scaling.
_CHANNEL_NORM: dict[str, dict] = {
    'combined_distance': {'scale': 5.0},         # z-scores, ~[-6, 6] → ~[-1.2, 1.2]
    'd_empirical':       {'scale': 5.0},         # dB change, ~[-10, 2] → ~[-2, 0.4]
    'fcf':               {},                      # already [0, 1]
    'slope':             {'scale': 0.6},          # radians, [0, ~1.2] → [0, ~2]
    'cell_counts':       {'log1p': True, 'scale': 5.0},  # heavy right skew → log1p/5
}


def _normalize_channel(arr: np.ndarray, var: str) -> np.ndarray:
    """Apply fixed normalization to a single channel array (in-place safe)."""
    cfg = _CHANNEL_NORM.get(var)
    if cfg is None or not cfg:
        return arr
    if cfg.get('log1p'):
        # sign-preserving log1p: sign(x) * log1p(|x|)
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return arr

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


def _clip_2d(da: xr.DataArray, geom) -> tuple[np.ndarray, np.ndarray]:
    """Clip a DataArray to a geometry and return (2d_array, mask).

    ``mask`` is True for pixels inside the polygon with finite values.
    Pixels outside the polygon are set to NaN by ``rio.clip``.
    """
    clipped = da.rio.clip([geom], all_touched=True, drop=True)
    arr = clipped.values.astype(float)
    mask = np.isfinite(arr)
    return arr, mask


# ── Spatial metric helpers ────────────────────────────────────────────────────

def _morans_i(arr: np.ndarray, mask: np.ndarray) -> float:
    """Moran's I spatial autocorrelation on a 2D masked grid (rook contiguity).

    Returns a float in [-1, +1]; NaN if fewer than 3 valid pixels or zero variance.
    """
    valid_idx = np.argwhere(mask)
    n = len(valid_idx)
    if n < 3:
        return np.nan

    vals = arr[mask]
    mean = vals.mean()
    var = vals.var()
    if var == 0:
        return np.nan

    # Build a fast lookup: (row, col) → index in vals
    idx_map = {}
    for i, (r, c) in enumerate(valid_idx):
        idx_map[(r, c)] = i

    # Rook neighbors (4-connected)
    W_sum = 0.0
    cross = 0.0
    for i, (r, c) in enumerate(valid_idx):
        zi = vals[i] - mean
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = idx_map.get((r + dr, c + dc))
            if j is not None:
                W_sum += 1.0
                cross += zi * (vals[j] - mean)

    if W_sum == 0:
        return np.nan
    return float((n / W_sum) * (cross / (n * var)))


def _hotspot_compactness(
    arr: np.ndarray, mask: np.ndarray, threshold_pct: float = 75,
) -> dict[str, float]:
    """Connected component analysis on above-threshold pixels.

    Returns ``n_clusters``, ``largest_frac``, and ``mean_cluster_dist``.
    """
    nan_keys = {'n_clusters': np.nan, 'largest_frac': np.nan, 'mean_cluster_dist': np.nan}
    vals = arr[mask]
    if vals.size < 3:
        return nan_keys

    threshold = np.nanpercentile(vals, threshold_pct)
    hot = (arr >= threshold) & mask
    if hot.sum() == 0:
        return nan_keys

    labeled, n_clusters = _label(hot)
    if n_clusters == 0:
        return nan_keys

    sizes = []
    centroids = []
    for lbl in range(1, n_clusters + 1):
        coords = np.argwhere(labeled == lbl)
        sizes.append(len(coords))
        centroids.append(coords.mean(axis=0))

    total_hot = sum(sizes)
    largest_frac = max(sizes) / total_hot

    if n_clusters <= 1:
        mean_dist = 0.0
    else:
        centroids = np.array(centroids)
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        mean_dist = float(np.mean(dists))

    return {
        'n_clusters': float(n_clusters),
        'largest_frac': float(largest_frac),
        'mean_cluster_dist': mean_dist,
    }


def _effective_radius(arr: np.ndarray, mask: np.ndarray, frac: float = 0.5) -> float:
    """Radius (in pixels) that contains ``frac`` of total signal, weighted by value.

    Uses signal-weighted centroid; returns NaN if no valid pixels.
    """
    valid_idx = np.argwhere(mask)
    if len(valid_idx) < 1:
        return np.nan

    vals = arr[mask]
    # Shift to non-negative weights
    w = vals - vals.min()
    total = w.sum()
    if total == 0:
        # Uniform values — use unweighted centroid
        centroid = valid_idx.mean(axis=0)
        dists = np.linalg.norm(valid_idx - centroid, axis=1)
        dists.sort()
        idx = int(np.ceil(frac * len(dists))) - 1
        return float(dists[max(idx, 0)])

    centroid = (valid_idx * w[:, np.newaxis]).sum(axis=0) / total
    dists = np.linalg.norm(valid_idx - centroid, axis=1)

    # Sort by distance and accumulate signal
    order = np.argsort(dists)
    cumw = np.cumsum(w[order])
    target = frac * total
    hit = np.searchsorted(cumw, target)
    hit = min(hit, len(dists) - 1)
    return float(dists[order[hit]])


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

    # 2D arrays for spatial metrics on per-track signal groups
    arrays_2d: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    per_track_keys: list[str] = []
    for group, pattern in _PER_TRACK_GROUPS.items():
        matching = [v for v in ds.data_vars if pattern.match(v)]
        log.debug("extract_track_features: %s → %d vars", group, len(matching))
        da = _pixel_max_da(ds, matching)
        if da is None:
            continue
        try:
            arrays[group] = _clip_arr(da, geom)
            arrays_2d[group] = _clip_2d(da, geom)
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
    for svar in static_vars:
        if svar not in arrays:
            continue
        for group in per_track_keys:
            _corr(arrays[svar], arrays[group], f'{svar}_x_{group}_corr')

    # ── Cross-correlations: per-track signal group pairs ──────────────────────
    for ga, gb in itertools.combinations(per_track_keys, 2):
        _corr(arrays[ga], arrays[gb], f'{ga}_x_{gb}_corr')

    # ── Spatial metrics on per-track signal groups ────────────────────────────
    _spatial_nan = {
        'morans_i': np.nan, 'n_clusters': np.nan, 'largest_frac': np.nan,
        'mean_cluster_dist': np.nan, 'eff_radius_50': np.nan,
    }
    for group in per_track_keys:
        if group not in arrays_2d:
            for k, v in _spatial_nan.items():
                features[f'{group}_{k}'] = v
            continue
        arr2d, mask2d = arrays_2d[group]
        try:
            features[f'{group}_morans_i'] = _morans_i(arr2d, mask2d)
        except Exception:
            features[f'{group}_morans_i'] = np.nan
        try:
            hc = _hotspot_compactness(arr2d, mask2d)
            features[f'{group}_n_clusters'] = hc['n_clusters']
            features[f'{group}_largest_frac'] = hc['largest_frac']
            features[f'{group}_mean_cluster_dist'] = hc['mean_cluster_dist']
        except Exception:
            features[f'{group}_n_clusters'] = np.nan
            features[f'{group}_largest_frac'] = np.nan
            features[f'{group}_mean_cluster_dist'] = np.nan
        try:
            features[f'{group}_eff_radius_50'] = _effective_radius(arr2d, mask2d)
        except Exception:
            features[f'{group}_eff_radius_50'] = np.nan

    # ── Track geometry features ───────────────────────────────────────────────
    try:
        features['track_area_m2'] = float(geom.area)
        features['track_perimeter_m'] = float(geom.length)
        perim = geom.length
        features['track_compactness'] = (
            float(4.0 * np.pi * geom.area / (perim ** 2)) if perim > 0 else np.nan
        )
        minx, miny, maxx, maxy = geom.bounds
        w = maxx - minx
        h = maxy - miny
        features['track_bbox_aspect_ratio'] = float(h / w) if w > 0 else np.nan
    except Exception:
        features['track_area_m2'] = np.nan
        features['track_perimeter_m'] = np.nan
        features['track_compactness'] = np.nan
        features['track_bbox_aspect_ratio'] = np.nan

    # Elevation range from DEM
    if 'dem' in arrays:
        dem_vals = arrays['dem'][np.isfinite(arrays['dem'])]
        features['track_elevation_range'] = (
            float(dem_vals.max() - dem_vals.min()) if dem_vals.size > 0 else np.nan
        )
    else:
        features['track_elevation_range'] = np.nan

    # ── Terrain character features ────────────────────────────────────────────
    if 'slope' in arrays:
        slope_vals = arrays['slope'][np.isfinite(arrays['slope'])]
        if slope_vals.size > 0:
            features['slope_range'] = float(slope_vals.max() - slope_vals.min())
            s_mean = slope_vals.mean()
            features['slope_cv'] = float(slope_vals.std() / s_mean) if s_mean > 0 else np.nan
        else:
            features['slope_range'] = np.nan
            features['slope_cv'] = np.nan
    else:
        features['slope_range'] = np.nan
        features['slope_cv'] = np.nan

    # ── Aspect circular statistics ────────────────────────────────────────────
    try:
        if 'aspect' in ds:
            aspect_vals = _clip_arr(ds['aspect'], geom)
            aspect_vals = aspect_vals[np.isfinite(aspect_vals)]
            if aspect_vals.size > 0:
                sin_vals = np.sin(aspect_vals)
                cos_vals = np.cos(aspect_vals)
                mean_sin = sin_vals.mean()
                mean_cos = cos_vals.mean()
                features['aspect_mean_resultant'] = float(
                    np.sqrt(mean_sin ** 2 + mean_cos ** 2)
                )
                features['aspect_mean_sin'] = float(mean_sin)
                features['aspect_mean_cos'] = float(mean_cos)
            else:
                features['aspect_mean_resultant'] = np.nan
                features['aspect_mean_sin'] = np.nan
                features['aspect_mean_cos'] = np.nan
        else:
            features['aspect_mean_resultant'] = np.nan
            features['aspect_mean_sin'] = np.nan
            features['aspect_mean_cos'] = np.nan
    except Exception:
        features['aspect_mean_resultant'] = np.nan
        features['aspect_mean_sin'] = np.nan
        features['aspect_mean_cos'] = np.nan

    # ── Runout character ──────────────────────────────────────────────────────
    if 'cell_counts' in arrays:
        cc_vals = arrays['cell_counts'][np.isfinite(arrays['cell_counts'])]
        features['cell_counts_range'] = (
            float(cc_vals.max() - cc_vals.min()) if cc_vals.size > 0 else np.nan
        )
    else:
        features['cell_counts_range'] = np.nan

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

    Channel order matches ``PATCH_CHANNELS`` (indices 0-7):
    combined_distance, d_empirical, fcf, slope, cell_counts,
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

    # Channels 0–4: raster data variables (normalized)
    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var not in ds.data_vars:
            continue
        arr = ds[var].sel(**clip_sel).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = _normalize_channel(arr, var)
        out[ch] = _zoom(arr, (zoom_y, zoom_x), order=1) if arr.shape != (size, size) else arr

    # Channel 5: northing — linear top=+1 → bottom=-1 (constant across all samples)
    north_vec = np.linspace(1.0, -1.0, size, dtype=np.float32)
    out[5] = np.broadcast_to(north_vec[:, np.newaxis], (size, size)).copy()

    # Channel 6: easting — linear left=-1 → right=+1
    east_vec = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    out[6] = np.broadcast_to(east_vec[np.newaxis, :], (size, size)).copy()

    # Channel 7: polygon mask at target resolution
    transform = _patch_transform(ref_da, size)
    out[7] = (~_geom_mask([geom], out_shape=(size, size), transform=transform,
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
