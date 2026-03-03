"""
Feature extraction for avalanche track polygons.

Produces flat dicts of floats suitable for tabular ML models (e.g. XGBoost).
For each track polygon, features are derived from three sources:

  - Static terrain variables (slope, DEM, FCF, runout, etc.) clipped from a
    reprojected xarray Dataset and summarised as mean/max/std/p75/p90.
  - Per-track SAR signal groups (empirical and ML-based backscatter change per
    polarisation), aggregated pixel-wise across orbits before clipping.
  - Track polygon geometry (area, compactness, aspect ratio, elevation range).

Cross-variable Pearson correlations are computed between every terrain variable
and SAR signal group pair, capturing spatial co-occurrence patterns such as
whether backscatter change is concentrated on steep or south-facing slopes.

Spatial structure metrics (Moran's I, hotspot compactness, effective signal
radius) are computed on the 2D max-aggregated SAR signal for each group.

Optionally, scene-level context features (from ``compute_scene_context``) can
be passed in to produce relative features comparing the track signal against
the scene background, flat-terrain baseline, and dominant aspect bin.

Typical usage
-------------
    scene_ctx = compute_scene_context(ds)
    features = extract_track_features(gdf.loc[42], ds, scene_ctx=scene_ctx)

Dependencies
------------
    sarvalanche.utils.raster_utils : clip_arr, clip_2d, pixel_agg_da,
                                     morans_i, hotspot_compactness, effective_radius
"""

import itertools
import logging
import re
import warnings

import numpy as np
import geopandas as gpd
import xarray as xr

from sarvalanche.utils.raster_utils import (
    clip_arr, clip_2d, pixel_agg_da,
    morans_i, hotspot_compactness, effective_radius,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STATIC_FEATURE_VARS: list[str] = [
    'fcf', 'cell_counts', 'slope', 'dem',
    'combined_distance', 'd_empirical', 'release_zones', 'runout_angle',
]

_PER_TRACK_GROUPS: dict[str, re.Pattern] = {
    'empirical_VV': re.compile(r'^p_\d+_VV_empirical$'),
    'empirical_VH': re.compile(r'^p_\d+_VH_empirical$'),
    'd_ml_VV':      re.compile(r'^d_\d+_VV_ml$'),
    'd_ml_VH':      re.compile(r'^d_\d+_VH_ml$'),
    'sigma_ml_VV':  re.compile(r'^sigma_\d+_VV_ml$'),
    'sigma_ml_VH':  re.compile(r'^sigma_\d+_VH_ml$'),
}

_STATS: dict[str, callable] = {
    'mean': np.nanmean,
    'max':  np.nanmax,
    'std':  np.nanstd,
    'p75':  lambda v: np.nanpercentile(v, 75),
    'p90':  lambda v: np.nanpercentile(v, 90),
}

_ASPECT_BINS: dict[str, tuple[float, float]] = {
    'N': (0.0,            np.pi / 4),
    'E': (np.pi / 4,      3 * np.pi / 4),
    'S': (3 * np.pi / 4,  5 * np.pi / 4),
    'W': (5 * np.pi / 4,  7 * np.pi / 4),
}
_FLAT_THRESHOLD: float = np.deg2rad(5)


# ── Aspect helpers ────────────────────────────────────────────────────────────

def _aspect_bin_mask(aspect: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'N':
        return (aspect < np.pi / 4) | (aspect >= 7 * np.pi / 4)
    lo, hi = _ASPECT_BINS[direction]
    return (aspect >= lo) & (aspect < hi)


def _dominant_aspect_bin(aspect_vals: np.ndarray) -> str:
    counts = {d: int(_aspect_bin_mask(aspect_vals, d).sum()) for d in ('N', 'E', 'S', 'W')}
    return max(counts, key=counts.get)


# ── Scene context ─────────────────────────────────────────────────────────────

def compute_scene_context(ds: xr.Dataset) -> dict[str, float]:
    """Compute scene-level context features from the full (unclipped) dataset.

    Returns scalar features summarising the whole scene's backscatter change
    and z-score distance, all prefixed with ``scene_``.
    """
    ctx: dict[str, float] = {}

    d_emp  = ds['d_empirical'].values.astype(float).ravel() if 'd_empirical' in ds else None
    c_dist = ds['combined_distance'].values.astype(float).ravel() if 'combined_distance' in ds else None
    slope  = ds['slope'].values.astype(float) if 'slope' in ds else None
    aspect = ds['aspect'].values.astype(float) if 'aspect' in ds else None

    flat_mask = (slope < _FLAT_THRESHOLD).ravel() if slope is not None else None

    def _stats2(vals, prefix, extra=()):
        finite = vals[np.isfinite(vals)] if vals is not None and vals.size else np.array([])
        ctx[f'{prefix}_mean'] = float(np.nanmean(finite)) if finite.size else np.nan
        ctx[f'{prefix}_std']  = float(np.nanstd(finite))  if finite.size else np.nan
        for pct in extra:
            ctx[f'{prefix}_p{pct}'] = float(np.nanpercentile(finite, pct)) if finite.size else np.nan

    if d_emp is not None and flat_mask is not None:
        _stats2(d_emp[flat_mask], 'scene_flat_d_empirical')
    else:
        ctx.update({'scene_flat_d_empirical_mean': np.nan, 'scene_flat_d_empirical_std': np.nan})

    if c_dist is not None and flat_mask is not None:
        flat_cd = c_dist[flat_mask & np.isfinite(c_dist)]
        ctx['scene_flat_combined_distance_mean'] = float(np.nanmean(flat_cd)) if flat_cd.size else np.nan
    else:
        ctx['scene_flat_combined_distance_mean'] = np.nan

    _stats2(d_emp,  'scene_d_empirical',      extra=(90,))
    _stats2(c_dist, 'scene_combined_distance')

    if d_emp is not None and slope is not None and aspect is not None:
        steep_mask = (slope >= _FLAT_THRESHOLD).ravel()
        for direction in ('N', 'E', 'S', 'W'):
            bin_mask = _aspect_bin_mask(aspect.ravel(), direction) & steep_mask & np.isfinite(d_emp)
            bin_vals = d_emp[bin_mask]
            ctx[f'scene_d_empirical_mean_{direction}'] = (
                float(np.nanmean(bin_vals)) if bin_vals.size else np.nan
            )
    else:
        for direction in ('N', 'E', 'S', 'W'):
            ctx[f'scene_d_empirical_mean_{direction}'] = np.nan

    return ctx


# ── Internal helpers ──────────────────────────────────────────────────────────

def _collect_arrays(
    geom,
    ds: xr.Dataset,
    static_vars: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Clip all needed arrays once.

    Static vars inserted first so area_pixels is always taken from a static var.
    Per-track groups: stacked once, all three aggs computed together.
    """
    arrays: dict[str, np.ndarray] = {}
    arrays_2d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_track_keys: list[str] = []

    for var in static_vars:
        if var not in ds:
            log.debug('_collect_arrays: %s not in dataset, skipping', var)
            continue
        try:
            arrays[var] = clip_arr(ds[var], geom)
        except Exception as exc:
            log.debug('_collect_arrays: clip failed for %s – %s', var, exc)

    for group, pattern in _PER_TRACK_GROUPS.items():
        matching = [v for v in ds.data_vars if pattern.match(v)]
        log.debug('_collect_arrays: %s → %d vars', group, len(matching))
        if not matching:
            continue

        ref = ds[matching[0]]
        stacked = np.stack([ds[v].values.astype(float) for v in matching], axis=0)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            agg_results = {
                'max':  np.nanmax(stacked, axis=0),
                'mean': np.nanmean(stacked, axis=0),
                'std':  np.nanstd(stacked, axis=0),
            }

        max_da = None
        for agg, agg_arr in agg_results.items():
            suffix = f'_{agg}' if agg != 'max' else ''
            key = f'{group}{suffix}'
            da = xr.DataArray(agg_arr, dims=ref.dims, coords=ref.coords).rio.write_crs(ref.rio.crs)
            try:
                arrays[key] = clip_arr(da, geom)
                if agg == 'max':
                    max_da = da
            except Exception as exc:
                log.debug('_collect_arrays: clip failed for %s – %s', key, exc)

        if max_da is not None and group in arrays:
            try:
                arrays_2d[group] = clip_2d(max_da, geom)
                per_track_keys.append(group)
            except Exception as exc:
                log.debug('_collect_arrays: 2d clip failed for %s – %s', group, exc)

    return arrays, arrays_2d, per_track_keys


def _compute_var_stats(
    arrays: dict[str, np.ndarray],
    static_vars: list[str],
) -> tuple[dict[str, float], int | None]:
    features: dict[str, float] = {}
    area_pixels: int | None = None
    static_set = set(static_vars)

    for name, arr in arrays.items():
        vals = arr[np.isfinite(arr)]
        if area_pixels is None and name in static_set:
            area_pixels = len(vals)
        if vals.size == 0:
            log.debug('_compute_var_stats: no valid pixels for %s', name)
            for stat in _STATS:
                features[f'{name}_{stat}'] = np.nan
            continue
        for stat, fn in _STATS.items():
            features[f'{name}_{stat}'] = float(fn(vals))

    return features, area_pixels


def _compute_correlations(
    arrays: dict[str, np.ndarray],
    static_vars: list[str],
    per_track_keys: list[str],
) -> dict[str, float]:
    features: dict[str, float] = {}

    def _corr(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3:
            return np.nan
        with np.errstate(invalid='ignore'):
            r = np.corrcoef(a[valid], b[valid])[0, 1]
        return float(r) if np.isfinite(r) else 0.0

    for svar in static_vars:
        if svar not in arrays:
            continue
        for group in per_track_keys:
            features[f'{svar}_x_{group}_corr'] = _corr(arrays[svar], arrays[group])

    for ga, gb in itertools.combinations(per_track_keys, 2):
        features[f'{ga}_x_{gb}_corr'] = _corr(arrays[ga], arrays[gb])

    return features


def _compute_spatial_metrics(
    arrays_2d: dict[str, tuple[np.ndarray, np.ndarray]],
    per_track_keys: list[str],
) -> dict[str, float]:
    features: dict[str, float] = {}
    nan_vals = {
        'morans_i': np.nan, 'n_clusters': np.nan, 'largest_frac': np.nan,
        'mean_cluster_dist': np.nan, 'eff_radius_50': np.nan,
    }

    for group in per_track_keys:
        if group not in arrays_2d:
            for k, v in nan_vals.items():
                features[f'{group}_{k}'] = v
            continue

        arr2d, mask2d = arrays_2d[group]

        try:
            features[f'{group}_morans_i'] = morans_i(arr2d, mask2d)
        except Exception:
            features[f'{group}_morans_i'] = np.nan

        try:
            hc = hotspot_compactness(arr2d, mask2d)
            features[f'{group}_n_clusters'] = hc['n_clusters']
            features[f'{group}_largest_frac'] = hc['largest_frac']
            features[f'{group}_mean_cluster_dist'] = hc['mean_cluster_dist']
        except Exception:
            features[f'{group}_n_clusters'] = np.nan
            features[f'{group}_largest_frac'] = np.nan
            features[f'{group}_mean_cluster_dist'] = np.nan

        try:
            features[f'{group}_eff_radius_50'] = effective_radius(arr2d, mask2d)
        except Exception:
            features[f'{group}_eff_radius_50'] = np.nan

    return features


def _compute_geometry_features(geom, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    features: dict[str, float] = {}

    try:
        area, perim = float(geom.area), float(geom.length)
        features['track_area_m2'] = area
        features['track_perimeter_m'] = perim
        features['track_compactness'] = float(4.0 * np.pi * area / perim ** 2) if perim > 0 else np.nan
        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx - minx, maxy - miny
        features['track_bbox_aspect_ratio'] = float(h / w) if w > 0 else np.nan
    except Exception as exc:
        log.warning('_compute_geometry_features: failed – %s', exc)
        features.update({'track_area_m2': np.nan, 'track_perimeter_m': np.nan,
                         'track_compactness': np.nan, 'track_bbox_aspect_ratio': np.nan})

    def _range(key):
        if key not in arrays:
            return np.nan
        v = arrays[key][np.isfinite(arrays[key])]
        return float(v.max() - v.min()) if v.size > 0 else np.nan

    features['track_elevation_range'] = _range('dem')
    features['cell_counts_range']      = _range('cell_counts')

    slope_vals = arrays['slope'][np.isfinite(arrays['slope'])] if 'slope' in arrays else np.array([])
    if slope_vals.size > 0:
        s_mean = slope_vals.mean()
        features['slope_range'] = float(slope_vals.max() - slope_vals.min())
        features['slope_cv'] = float(slope_vals.std() / s_mean) if s_mean > 0 else np.nan
    else:
        features['slope_range'] = np.nan
        features['slope_cv'] = np.nan

    return features


def _compute_aspect_features(
    ds: xr.Dataset, geom,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Returns (features, aspect_vals) — aspect_vals reused in scene-relative step."""
    nan_aspect = {'aspect_mean_resultant': np.nan,
                  'aspect_mean_sin': np.nan, 'aspect_mean_cos': np.nan}
    if 'aspect' not in ds:
        return nan_aspect, None
    try:
        aspect_vals = clip_arr(ds['aspect'], geom)
        aspect_vals = aspect_vals[np.isfinite(aspect_vals)]
        if aspect_vals.size == 0:
            return nan_aspect, None
        sin_mean = float(np.sin(aspect_vals).mean())
        cos_mean = float(np.cos(aspect_vals).mean())
        return {
            'aspect_mean_resultant': float(np.sqrt(sin_mean ** 2 + cos_mean ** 2)),
            'aspect_mean_sin': sin_mean,
            'aspect_mean_cos': cos_mean,
        }, aspect_vals
    except Exception as exc:
        log.warning('_compute_aspect_features: failed – %s', exc)
        return nan_aspect, None


def _compute_scene_relative_features(
    features: dict[str, float],
    scene_ctx: dict[str, float],
    aspect_vals: np.ndarray | None,
) -> dict[str, float]:
    relative = dict(scene_ctx)
    track_d_mean  = features.get('d_empirical_mean', np.nan)
    track_cd_mean = features.get('combined_distance_mean', np.nan)

    relative['d_empirical_mean_vs_scene'] = track_d_mean  - scene_ctx.get('scene_d_empirical_mean', np.nan)
    relative['d_empirical_mean_vs_flat']  = track_d_mean  - scene_ctx.get('scene_flat_d_empirical_mean', np.nan)
    relative['combined_distance_mean_vs_scene'] = track_cd_mean - scene_ctx.get('scene_combined_distance_mean', np.nan)

    if aspect_vals is not None and aspect_vals.size > 0:
        dom_bin = _dominant_aspect_bin(aspect_vals)
        relative['d_empirical_mean_vs_aspect'] = (
            track_d_mean - scene_ctx.get(f'scene_d_empirical_mean_{dom_bin}', np.nan)
        )
    else:
        relative['d_empirical_mean_vs_aspect'] = np.nan

    return relative


# ── Public API ────────────────────────────────────────────────────────────────

def extract_track_features(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    static_vars: list[str] = STATIC_FEATURE_VARS,
    scene_ctx: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract aggregate statistics and correlations for a track polygon.

    Parameters
    ----------
    row         : single track; .geometry must match ds CRS
    ds          : reprojected dataset; must contain static_vars
    static_vars : terrain/physical variables to aggregate directly
    scene_ctx   : pre-computed output of compute_scene_context()

    Returns
    -------
    dict[str, float]
    """
    geom = row.geometry
    arrays, arrays_2d, per_track_keys = _collect_arrays(geom, ds, static_vars)

    features, area_pixels = _compute_var_stats(arrays, static_vars)
    features.update(_compute_correlations(arrays, static_vars, per_track_keys))
    features.update(_compute_spatial_metrics(arrays_2d, per_track_keys))
    features.update(_compute_geometry_features(geom, arrays))

    aspect_features, aspect_vals = _compute_aspect_features(ds, geom)
    features.update(aspect_features)

    if scene_ctx is not None:
        features.update(_compute_scene_relative_features(features, scene_ctx, aspect_vals))

    features['area_pixels'] = float(area_pixels) if area_pixels is not None else np.nan
    return features