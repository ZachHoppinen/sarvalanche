import itertools
import logging
import re

import numpy as np
import geopandas as gpd
import xarray as xr

log = logging.getLogger(__name__)

# Static 2D terrain/physical variables — raw values, no probability transforms
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
