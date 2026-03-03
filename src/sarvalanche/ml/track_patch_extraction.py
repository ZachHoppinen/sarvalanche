"""
Raster patch extraction for avalanche track polygons.

Produces fixed-size (C, H, W) float32 tensors suitable for CNN input.
Each patch is centred on a track polygon's bounding box (plus a configurable
buffer) and resampled to a square grid via bilinear zoom.

Channels are defined by PATCH_CHANNELS and cover eight raster data variables
(backscatter change, terrain, runout, water mask), two synthetic coordinate
channels (northing, easting), and a binary track polygon mask. All data
channels are normalised to a consistent scale before output; see _CHANNEL_NORM
for per-channel scaling constants.

Three public functions are provided:

  - ``extract_track_patch``             : patch only (inference / visualisation)
  - ``extract_track_patch_with_target`` : patch + soft segmentation target
                                          (training the CNN segmentation head)
  - ``aggregate_seg_features``          : collapses a (H, W) segmentation
                                          probability map back to scalar features
                                          for use alongside tabular models

Pass ``row=None`` to either extraction function to operate on the full dataset
extent rather than a single track, with the track mask channel set to all-ones.
"""

import logging

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask as _geom_mask
from rasterio.transform import from_bounds as _from_bounds
from scipy.ndimage import zoom as _zoom

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_CHANNELS: list[str] = [
    'combined_distance',  # 0 — ML z-score distance (signed, std devs)
    'd_empirical',        # 1 — combined backscatter change (dB)
    'fcf',                # 2 — forest cover fraction
    'slope',              # 3 — terrain slope (radians)
    'cell_counts',        # 4 — FlowPy runout cell counts
    'release_zones',      # 5 — binary release zone mask
    'runout_angle',       # 6 — debris flow direction (radians)
    'water_mask',         # 7 — binary water body mask
    'northing',           # 8 — relative y position [-1 (bottom), +1 (top)]
    'easting',            # 9 — relative x position [-1 (left),  +1 (right)]
    'track_mask',         # 10 — 1 inside track polygon, 0 outside
]
N_PATCH_CHANNELS: int = len(PATCH_CHANNELS)
TRACK_MASK_CHANNEL: int = PATCH_CHANNELS.index('track_mask')

_PATCH_DATA_VARS: list[str] = [
    'combined_distance', 'd_empirical', 'fcf', 'slope', 'cell_counts',
    'release_zones', 'runout_angle', 'water_mask',
]
_TARGET_VAR: str = 'unmasked_p_target'

_NORTHING_CH: int = len(_PATCH_DATA_VARS)
_EASTING_CH:  int = len(_PATCH_DATA_VARS) + 1

_CHANNEL_NORM: dict[str, dict] = {
    'combined_distance': {'scale': 5.0},
    'd_empirical':       {'scale': 5.0},
    'fcf':               {},
    'slope':             {'scale': 0.6},
    'cell_counts':       {'log1p': True, 'scale': 5.0},
    'runout_angle':      {'scale': np.pi},
}

_SEG_FEATURE_NAMES: list[str] = [
    'seg_mean', 'seg_max', 'seg_p75', 'seg_p90', 'seg_p95', 'seg_frac_above_05',
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_channel(arr: np.ndarray, var: str) -> np.ndarray:
    cfg = _CHANNEL_NORM.get(var)
    if not cfg:
        return arr
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return arr


def _patch_transform(ref_da: xr.DataArray, size: int):
    x_vals, y_vals = ref_da.x.values, ref_da.y.values
    dx = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 30.0
    dy = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 30.0
    return _from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2,
        size, size,
    )


def _clip_sel(geom, buffer: float) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    return dict(
        x=slice(minx - buffer, maxx + buffer),
        y=slice(maxy + buffer, miny - buffer),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_track_patch(
    row: gpd.GeoSeries | None,
    ds: xr.Dataset,
    size: int = 64,
    buffer: float = 1000.0,
) -> np.ndarray:
    """Extract a (C, size, size) float32 patch centred on the track bbox.

    Pass ``row=None`` for full-scene mode (track_mask becomes all-ones).
    Channel order matches PATCH_CHANNELS.
    """
    geom = row.geometry if row is not None else None
    sel = _clip_sel(geom, buffer) if geom is not None else {}

    ref_var = next((v for v in _PATCH_DATA_VARS if v in ds.data_vars), None)
    if ref_var is None:
        log.warning('extract_track_patch: none of %s found in dataset', _PATCH_DATA_VARS)
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    ref_da = ds[ref_var].sel(**sel)
    H, W = ref_da.shape
    if H < 2 or W < 2:
        log.debug('extract_track_patch: clipped region too small (%d×%d)', H, W)
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    out = np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)
    zoom_y, zoom_x = size / H, size / W

    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var not in ds.data_vars:
            continue
        arr = np.nan_to_num(ds[var].sel(**sel).values.astype(np.float32), nan=0.0)
        arr = _normalize_channel(arr, var)
        out[ch] = _zoom(arr, (zoom_y, zoom_x), order=1) if arr.shape != (size, size) else arr

    north_vec = np.linspace(1.0, -1.0, size, dtype=np.float32)
    out[_NORTHING_CH] = np.broadcast_to(north_vec[:, np.newaxis], (size, size)).copy()
    east_vec = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    out[_EASTING_CH] = np.broadcast_to(east_vec[np.newaxis, :], (size, size)).copy()

    if geom is not None:
        out[TRACK_MASK_CHANNEL] = (~_geom_mask(
            [geom], out_shape=(size, size),
            transform=_patch_transform(ref_da, size), all_touched=True,
        )).astype(np.float32)
    else:
        out[TRACK_MASK_CHANNEL] = 1.0

    return out


def extract_track_patch_with_target(
    row: gpd.GeoSeries | None,
    ds: xr.Dataset,
    size: int = 64,
    buffer: float = 1000.0,
    debris_shapes: gpd.GeoDataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract patch + (1, size, size) soft segmentation target.

    Target comes from ``unmasked_p_target``; all-zeros if not present.
    ``debris_shapes`` polygons are blended in via element-wise max when provided.
    """
    patch = extract_track_patch(row, ds, size=size, buffer=buffer)
    target = np.zeros((1, size, size), dtype=np.float32)

    if _TARGET_VAR not in ds.data_vars:
        return patch, target

    geom = row.geometry if row is not None else None
    sel = _clip_sel(geom, buffer) if geom is not None else {}

    ref_var = next((v for v in _PATCH_DATA_VARS if v in ds.data_vars), None)
    ref_da = ds[ref_var].sel(**sel) if ref_var else None

    arr = np.nan_to_num(ds[_TARGET_VAR].sel(**sel).values.astype(np.float32), nan=0.0)
    H, W = arr.shape
    if H >= 2 and W >= 2:
        zoom_y, zoom_x = size / H, size / W
        target[0] = _zoom(arr, (zoom_y, zoom_x), order=1) if arr.shape != (size, size) else arr

    if debris_shapes is not None and not debris_shapes.empty and ref_da is not None:
        rH, rW = ref_da.shape
        if rH >= 2 and rW >= 2:
            shape_mask = (~_geom_mask(
                list(debris_shapes.geometry), out_shape=(size, size),
                transform=_patch_transform(ref_da, size), all_touched=True,
            )).astype(np.float32)
            target[0] = np.maximum(target[0], shape_mask)

    return patch, target


def aggregate_seg_features(
    seg_map: np.ndarray,
    track_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate a (H, W) segmentation probability map into scalar features."""
    vals = seg_map[track_mask > 0.5] if track_mask is not None else seg_map.ravel()
    if vals.size == 0:
        return {k: np.nan for k in _SEG_FEATURE_NAMES}
    return {
        'seg_mean':          float(np.nanmean(vals)),
        'seg_max':           float(np.nanmax(vals)),
        'seg_p75':           float(np.nanpercentile(vals, 75)),
        'seg_p90':           float(np.nanpercentile(vals, 90)),
        'seg_p95':           float(np.nanpercentile(vals, 95)),
        'seg_frac_above_05': float((vals > 0.5).sum() / len(vals)),
    }