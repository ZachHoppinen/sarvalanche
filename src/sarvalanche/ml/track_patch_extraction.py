"""
Dual-scale raster patch extraction for avalanche track polygons.

Produces paired (context, local_tiles) tensors for CNN input:

  - Context patch  : full track bbox zoomed to (C, 64, 64). Captures track
                     morphology, terrain structure, and regional SAR signal.
  - Local tiles    : fixed physical resolution (C, 64, 64) tiles sliding
                     across the track at native WGS84 resolution. Each tile
                     includes within-track position channels so the model
                     knows where in the track it is looking.

Channel order is defined by PATCH_CHANNELS and is identical for both branches.
Normalisation constants are defined in _CHANNEL_NORM.

Public API
----------
  extract_context_patch(row, ds, size, buffer_deg, src_crs)
      → (C, size, size) float32

  extract_local_tiles(row, ds, size, resolution_deg, overlap, src_crs)
      → (N, C, size, size) float32,  tile_positions (N, 4) float32

  extract_dual_scale_patch(row, ds, size, buffer_deg, resolution_deg, overlap, src_crs)
      → DualScalePatch(context, local_tiles, tile_positions)

  extract_dual_scale_with_target(row, ds, ..., debris_shapes)
      → DualScalePatch, targets (N+1, 1, size, size)

  aggregate_seg_features(seg_map, track_mask)
      → dict[str, float]
"""

import logging
from dataclasses import dataclass

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask as _geom_mask
from rasterio.transform import from_bounds as _from_bounds
from skimage.transform import resize as _resize
from shapely.geometry import box

from sarvalanche.ml.track_features import reproject_geom
from sarvalanche.utils.raster_utils import _y_slice

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

# ~1km buffer in degrees at ~44°N latitude
# _DEFAULT_BUFFER_DEG: float = 0.009
_DEFAULT_BUFFER_DEG: float = 0.0

# ~10m pixels in degrees at ~44°N
# _DEFAULT_RESOLUTION_DEG: float = 0.0001

def _ds_resolution(ds: xr.Dataset) -> float:
    """Get the dataset's native resolution in degrees (mean of x and y spacing)."""
    x_res = abs(float(ds.x.values[1] - ds.x.values[0]))
    y_res = abs(float(ds.y.values[1] - ds.y.values[0]))
    return (x_res + y_res) / 2

# ── Channel definitions ───────────────────────────────────────────────────────

PATCH_CHANNELS: list[str] = [
    'combined_distance',  # 0 — ML z-score distance
    'd_empirical',        # 1 — backscatter change (dB)
    'fcf',                # 2 — forest cover fraction
    'slope',              # 3 — terrain slope (radians)
    'cell_counts',        # 4 — FlowPy runout cell counts
    'release_zones',      # 5 — binary release zone mask
    'runout_angle',       # 6 — debris flow direction (radians)
    'water_mask',         # 7 — binary water body mask
    'northing',           # 8 — position within full track [-1 (bottom), +1 (top)]
    'easting',            # 9 — position within full track [-1 (left),  +1 (right)]
    'track_mask',         # 10 — 1 inside polygon, 0 outside
]

N_PATCH_CHANNELS: int = len(PATCH_CHANNELS)
TRACK_MASK_CHANNEL: int = PATCH_CHANNELS.index('track_mask')

_PATCH_DATA_VARS: list[str] = [
    'combined_distance', 'd_empirical', 'fcf', 'slope',
    'cell_counts', 'release_zones', 'runout_angle', 'water_mask',
]
_TARGET_VAR: str = 'unmasked_p_target'
_NORTHING_CH: int = PATCH_CHANNELS.index('northing')
_EASTING_CH:  int = PATCH_CHANNELS.index('easting')

_CHANNEL_NORM: dict[str, dict] = {
    'combined_distance': {'scale': 5.0},
    'd_empirical':       {'scale': 5.0},
    'fcf':               {'scale': 100.0},
    'slope':             {'scale': 0.6},
    'cell_counts':       {'log1p': True, 'scale': 5.0},
    'runout_angle':      {'scale': np.pi},
}

_SEG_FEATURE_NAMES: list[str] = [
    'seg_mean', 'seg_max', 'seg_p75', 'seg_p90', 'seg_p95', 'seg_frac_above_05',
]


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class DualScalePatch:
    """Output of extract_dual_scale_patch.

    Attributes
    ----------
    context : np.ndarray of shape (C, size, size)
        Full track zoomed to fixed size. Northing/easting encode position
        within the full track bbox.
    local_tiles : np.ndarray of shape (N, C, size, size)
        Fixed physical resolution tiles. Northing/easting encode position
        within the full track bbox, not the tile bbox.
    tile_positions : np.ndarray of shape (N, 4)
        Each row is (norm_x_min, norm_y_min, norm_x_max, norm_y_max) giving
        the tile's location within the full track bbox, normalised to [0, 1].
    """
    context: np.ndarray
    local_tiles: np.ndarray
    tile_positions: np.ndarray


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_resize(arr: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    """Resize array to exactly (size, size) using skimage."""
    if arr.shape == (size, size):
        return arr
    return _resize(
        arr, (size, size),
        order=order,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(arr.dtype)


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


def _patch_transform(da_local: xr.DataArray, size: int):
    """Affine transform mapping (size, size) pixel grid to da_local coordinates."""
    x_vals = da_local.x.values
    y_vals = da_local.y.values
    dx = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 1e-4
    dy = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 1e-4
    return _from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2,
        size, size,
    )


def _track_mask_channel(geom, da_local: xr.DataArray, size: int) -> np.ndarray:
    """Rasterize track polygon to (size, size) binary mask."""
    transform = _patch_transform(da_local, size)
    return (~_geom_mask(
        [geom], out_shape=(size, size),
        transform=transform, all_touched=True,
    )).astype(np.float32)


def _position_channels(
    tile_minx: float, tile_miny: float,
    tile_maxx: float, tile_maxy: float,
    track_minx: float, track_miny: float,
    track_maxx: float, track_maxy: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Northing/easting channels encoding position within the full track bbox.

    Values are normalised so that the track bbox spans [-1, +1] in both axes.
    Tiles partially outside the track bbox will have values outside [-1, +1].
    """
    track_w = track_maxx - track_minx
    track_h = track_maxy - track_miny

    east_left  = 2 * (tile_minx - track_minx) / track_w - 1
    east_right = 2 * (tile_maxx - track_minx) / track_w - 1
    north_top  = 2 * (tile_maxy - track_miny) / track_h - 1
    north_bot  = 2 * (tile_miny - track_miny) / track_h - 1

    northing = np.linspace(north_top, north_bot, size, dtype=np.float32)
    easting  = np.linspace(east_left, east_right, size, dtype=np.float32)

    return (
        np.broadcast_to(northing[:, np.newaxis], (size, size)).copy(),
        np.broadcast_to(easting[np.newaxis, :], (size, size)).copy(),
    )


def _extract_patch_arrays(
    ds_local: xr.Dataset,
    geom,
    track_bounds: tuple,
    size: int,
    zoom: bool,
) -> np.ndarray:
    """Extract (C, size, size) patch from a pre-sliced ds_local.

    Parameters
    ----------
    ds_local : bbox-clipped dataset
    geom : polygon in dataset CRS, or None for full-scene mode
    track_bounds : (minx, miny, maxx, maxy) of full track bbox for position channels
    size : output patch size
    zoom : if True resize to size×size, if False assume ds_local is already size×size
    """
    ref_var = next((v for v in _PATCH_DATA_VARS if v in ds_local.data_vars), None)
    if ref_var is None:
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    ref_da = ds_local[ref_var]
    H, W = ref_da.shape
    if H < 2 or W < 2:
        return np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    out = np.zeros((N_PATCH_CHANNELS, size, size), dtype=np.float32)

    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var not in ds_local.data_vars:
            continue
        arr = np.nan_to_num(ds_local[var].values.astype(np.float32), nan=0.0)
        arr = _normalize_channel(arr, var)
        out[ch] = _safe_resize(arr, size, order=1) if zoom else arr

    # Position channels relative to full track bbox
    tile_x = ref_da.x.values
    tile_y = ref_da.y.values
    tile_bounds = (
        float(tile_x.min()), float(tile_y.min()),
        float(tile_x.max()), float(tile_y.max()),
    )
    north_ch, east_ch = _position_channels(*tile_bounds, *track_bounds, size)
    out[_NORTHING_CH] = north_ch
    out[_EASTING_CH]  = east_ch

    # Track mask
    if geom is not None:
        mask_full = _track_mask_channel(geom, ref_da, H)
        out[TRACK_MASK_CHANNEL] = _safe_resize(mask_full, size, order=0) > 0.5
    else:
        out[TRACK_MASK_CHANNEL] = 1.0

    return out


def _bbox_sel(ds: xr.Dataset, minx, miny, maxx, maxy, buffer: float = 0.0):
    """Slice dataset to bbox with optional buffer."""
    ref = next(iter(ds.data_vars.values()))
    return ds.sel(
        x=slice(minx - buffer, maxx + buffer),
        y=_y_slice(ref, miny - buffer, maxy + buffer),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_context_patch(
    row: gpd.GeoSeries | None,
    ds: xr.Dataset,
    size: int = 64,
    buffer_deg: float = _DEFAULT_BUFFER_DEG,
    src_crs=None,
) -> np.ndarray:
    """Extract a (C, size, size) context patch zoomed from the full track bbox.

    Parameters
    ----------
    row : track row; pass None for full-scene mode
    ds : dataset in WGS84
    size : output patch size in pixels
    buffer_deg : buffer around track bbox in degrees (~0.009° ≈ 1km at 44°N)
    src_crs : CRS of row.geometry; if differs from ds.rio.crs, geom is reprojected

    Returns
    -------
    np.ndarray of shape (C, size, size), dtype float32
    """
    if row is None:
        track_bounds = (
            float(ds.x.min()), float(ds.y.min()),
            float(ds.x.max()), float(ds.y.max()),
        )
        return _extract_patch_arrays(ds, None, track_bounds, size, zoom=True)

    geom = row.geometry
    if src_crs is not None and ds.rio.crs is not None and src_crs != ds.rio.crs:
        geom = reproject_geom(geom, src_crs, ds.rio.crs)

    minx, miny, maxx, maxy = geom.bounds
    track_bounds = (minx, miny, maxx, maxy)
    ds_local = _bbox_sel(ds, minx, miny, maxx, maxy, buffer=buffer_deg)

    return _extract_patch_arrays(ds_local, geom, track_bounds, size, zoom=True)


def extract_local_tiles(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    size: int = 64,
    resolution_deg: float | None = None,
    overlap: float = 0.5,
    src_crs=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed physical resolution tiles across the track bbox.

    Parameters
    ----------
    row : track row
    ds : dataset in WGS84
    size : tile size in pixels
    resolution_deg : physical resolution per pixel in degrees (~10m at 44°N)
    overlap : fractional overlap between adjacent tiles (0=no overlap, 0.5=50%)
    src_crs : CRS of row.geometry

    Returns
    -------
    tiles : np.ndarray of shape (N, C, size, size), dtype float32
    tile_positions : np.ndarray of shape (N, 4)
        Each row: (norm_x_min, norm_y_min, norm_x_max, norm_y_max) in [0, 1]
        relative to the track bbox.
    """
    geom = row.geometry
    if src_crs is not None and ds.rio.crs is not None and src_crs != ds.rio.crs:
        geom = reproject_geom(geom, src_crs, ds.rio.crs)
    if resolution_deg is None:
        resolution_deg = _ds_resolution(ds)

    minx, miny, maxx, maxy = geom.bounds
    track_bounds = (minx, miny, maxx, maxy)
    track_w = maxx - minx
    track_h = maxy - miny

    tile_w = size * resolution_deg
    tile_h = size * resolution_deg
    stride_w = tile_w * (1 - overlap)
    stride_h = tile_h * (1 - overlap)

    x_starts = np.arange(minx, maxx, stride_w)
    y_starts = np.arange(miny, maxy, stride_h)

    tiles = []
    positions = []

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = x0 + tile_w
            y1 = y0 + tile_h

            if not geom.intersects(box(x0, y0, x1, y1)):
                continue

            ds_local = _bbox_sel(ds, x0, y0, x1, y1)
            if ds_local.sizes.get('x', 0) < 2 or ds_local.sizes.get('y', 0) < 2:
                continue

            patch = _extract_patch_arrays(ds_local, geom, track_bounds, size, zoom=True)
            tiles.append(patch)

            positions.append([
                (x0 - minx) / track_w,
                (y0 - miny) / track_h,
                (x1 - minx) / track_w,
                (y1 - miny) / track_h,
            ])

    if not tiles:
        log.warning('extract_local_tiles: no tiles generated for track %s', getattr(row, 'name', '?'))
        return (
            np.zeros((0, N_PATCH_CHANNELS, size, size), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    return (
        np.stack(tiles, axis=0).astype(np.float32),
        np.array(positions, dtype=np.float32),
    )


def extract_dual_scale_patch(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    size: int = 64,
    buffer_deg: float = _DEFAULT_BUFFER_DEG,
    resolution_deg: float | None = None,
    overlap: float = 0.5,
    src_crs=None,
) -> DualScalePatch:
    """Extract paired context + local tile patches for dual-scale CNN input.

    Parameters
    ----------
    row : track row
    ds : dataset in WGS84
    size : patch size for both branches
    buffer_deg : buffer around track bbox for context patch
    resolution_deg : physical resolution per pixel for local tiles (~0.0001° ≈ 10m)
    overlap : tile overlap fraction
    src_crs : CRS of row.geometry

    Returns
    -------
    DualScalePatch with context (C,H,W), local_tiles (N,C,H,W), tile_positions (N,4)
    """
    context = extract_context_patch(
        row, ds, size=size, buffer_deg=buffer_deg, src_crs=src_crs
    )
    local_tiles, tile_positions = extract_local_tiles(
        row, ds, size=size, resolution_deg=resolution_deg,
        overlap=overlap, src_crs=src_crs,
    )
    return DualScalePatch(
        context=context,
        local_tiles=local_tiles,
        tile_positions=tile_positions,
    )


def extract_dual_scale_with_target(
    row: gpd.GeoSeries,
    ds: xr.Dataset,
    size: int = 64,
    buffer_deg: float = _DEFAULT_BUFFER_DEG,
    resolution_deg: float | None = None,
    overlap: float = 0.5,
    src_crs=None,
    debris_shapes: gpd.GeoDataFrame | None = None,
) -> tuple[DualScalePatch, np.ndarray]:
    """Extract dual-scale patch + segmentation targets for training.

    Returns
    -------
    patch : DualScalePatch
    targets : np.ndarray of shape (N+1, 1, size, size)
        Index 0 is the context target; indices 1..N are per-tile targets.
        Values come from unmasked_p_target, blended with debris_shapes via
        element-wise max when provided.
    """
    patch = extract_dual_scale_patch(
        row, ds, size=size, buffer_deg=buffer_deg,
        resolution_deg=resolution_deg, overlap=overlap, src_crs=src_crs,
    )
    n_tiles = len(patch.local_tiles)
    targets = np.zeros((n_tiles + 1, 1, size, size), dtype=np.float32)

    if _TARGET_VAR not in ds.data_vars:
        return patch, targets

    geom = row.geometry
    if src_crs is not None and ds.rio.crs is not None and src_crs != ds.rio.crs:
        geom = reproject_geom(geom, src_crs, ds.rio.crs)
    if resolution_deg is None:
        resolution_deg = _ds_resolution(ds)


    minx, miny, maxx, maxy = geom.bounds

    def _extract_target(ds_local: xr.Dataset) -> np.ndarray:
        if _TARGET_VAR not in ds_local.data_vars:
            return np.zeros((1, size, size), dtype=np.float32)
        arr = np.nan_to_num(
            ds_local[_TARGET_VAR].values.astype(np.float32), nan=0.0
        )
        H, W = arr.shape
        if H < 2 or W < 2:
            return np.zeros((1, size, size), dtype=np.float32)
        t = _safe_resize(arr, size, order=1)
        if debris_shapes is not None and not debris_shapes.empty:
            ref_var = next((v for v in _PATCH_DATA_VARS if v in ds_local.data_vars), None)
            if ref_var is not None:
                shape_mask = (~_geom_mask(
                    list(debris_shapes.geometry),
                    out_shape=(size, size),
                    transform=_patch_transform(ds_local[ref_var], size),
                    all_touched=True,
                )).astype(np.float32)
                t = np.maximum(t, shape_mask)
        return t[np.newaxis]

    # Context target
    targets[0] = _extract_target(_bbox_sel(ds, minx, miny, maxx, maxy, buffer=buffer_deg))

    # Per-tile targets — regenerate tile grid to match extract_local_tiles
    tile_w = size * resolution_deg
    tile_h = size * resolution_deg
    stride_w = tile_w * (1 - overlap)
    stride_h = tile_h * (1 - overlap)

    tile_idx = 1
    for y0 in np.arange(miny, maxy, stride_h):
        for x0 in np.arange(minx, maxx, stride_w):
            if tile_idx >= n_tiles + 1:
                break
            x1, y1 = x0 + tile_w, y0 + tile_h
            if not geom.intersects(box(x0, y0, x1, y1)):
                continue
            ds_local = _bbox_sel(ds, x0, y0, x1, y1)
            if ds_local.sizes.get('x', 0) < 2 or ds_local.sizes.get('y', 0) < 2:
                continue
            targets[tile_idx] = _extract_target(ds_local)
            tile_idx += 1

    return patch, targets


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