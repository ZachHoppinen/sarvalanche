"""Fetch HRRR 2m temperature snapshots aligned to SAR overpass times.

For each SAR acquisition timestamp, downloads the nearest HRRR cycle's
2m temperature field, resamples it to the SAR grid, and applies an
elevation-based lapse-rate correction using the DEM.

Output variables (per SAR time step):
  - ``t2m``: 2m temperature in °C, lapse-rate adjusted to DEM elevation.

Speed optimisations
-------------------
- Batch download via ``FastHerbie`` (threaded parallel downloads).
- The KD-tree for HRRR→SAR resampling is built once and reused.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAPSE_RATE = -6.5e-3  # °C per metre (standard atmosphere)
KELVIN_OFFSET = -273.15

# HRRR-AK cycles every 3 h; CONUS every 1 h.
_CYCLE_HOURS: dict[str, list[int]] = {
    'hrrrak': list(range(0, 24, 3)),  # [0, 3, 6, …, 21]
    'hrrr': list(range(0, 24)),  # [0, 1, 2, …, 23]
}

# Rough Alaska bounding box (lat/lon) — anything inside uses hrrrak.
_AK_LAT_MIN, _AK_LAT_MAX = 51.0, 72.0
_AK_LON_MIN, _AK_LON_MAX = -180.0, -129.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_hrrr_model(ds: xr.Dataset) -> str:
    """Choose ``"hrrrak"`` or ``"hrrr"`` based on the dataset's bounding box.

    Reprojects the grid corners to EPSG:4326 and checks whether the
    centroid falls inside the Alaska domain.
    """
    crs = ds.rio.crs
    if crs is None:
        raise ValueError('Dataset must have a CRS to auto-detect HRRR model')

    y_vals = ds.y.values
    x_vals = ds.x.values
    cy = float(y_vals.mean())
    cx = float(x_vals.mean())

    crs_obj = CRS(crs)
    if not crs_obj.is_geographic:
        transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(4326), always_xy=True)
        cx, cy = transformer.transform(cx, cy)

    if _AK_LAT_MIN <= cy <= _AK_LAT_MAX and _AK_LON_MIN <= cx <= _AK_LON_MAX:
        log.info('Auto-detected HRRR model: hrrrak (centroid lat=%.2f, lon=%.2f)', cy, cx)
        return 'hrrrak'
    log.info('Auto-detected HRRR model: hrrr (centroid lat=%.2f, lon=%.2f)', cy, cx)
    return 'hrrr'


def nearest_cycle_hour(timestamp: pd.Timestamp, model: str = 'hrrrak') -> int:
    """Return the HRRR cycle hour closest to *timestamp* (UTC).

    Parameters
    ----------
    timestamp : pd.Timestamp
        SAR acquisition time (tz-naive assumed UTC, or tz-aware).
    model : str
        ``"hrrrak"`` (3-hourly) or ``"hrrr"`` (hourly).

    Returns:
    -------
    int
        The cycle hour (0–23) nearest to *timestamp*.
    """
    hours = _CYCLE_HOURS.get(model)
    if hours is None:
        raise ValueError(f'Unknown HRRR model {model!r}; expected one of {list(_CYCLE_HOURS)}')

    frac_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    # Find closest cycle hour (wrap-around aware)
    diffs = [min(abs(frac_hour - h), 24 - abs(frac_hour - h)) for h in hours]
    return hours[int(np.argmin(diffs))]


def fetch_t2m_batch(
    cycle_datetimes: list[str],
    model: str = 'hrrrak',
    max_threads: int = 10,
) -> xr.Dataset | None:
    """Batch-download HRRR 2m temperature for multiple cycle times.

    Uses ``FastHerbie`` for parallel threaded downloads.

    Parameters
    ----------
    cycle_datetimes : list[str]
        Datetime strings like ``"2025-01-15 06:00"`` (one per HRRR cycle).
    model : str
        ``"hrrrak"`` or ``"hrrr"``.
    max_threads : int
        Number of download threads for FastHerbie.

    Returns:
    -------
    xr.Dataset or None
        Combined dataset with dims ``(time, y, x)`` and variables
        ``t2m`` (Kelvin), ``latitude``, ``longitude``.
        Returns *None* if all downloads fail.
    """
    from herbie import FastHerbie

    try:
        FH = FastHerbie(
            cycle_datetimes,
            model=model,
            product='sfc',
            fxx=[0],
            verbose=False,
            max_threads=max_threads,
        )
        ds = FH.xarray('TMP:2 m above ground', max_threads=max_threads)
        return ds
    except Exception as exc:
        log.warning('FastHerbie batch fetch failed (%s): %s', model, exc)
        return None


# ---------------------------------------------------------------------------
# Resampling (KD-tree)
# ---------------------------------------------------------------------------


def build_kdtree(hrrr_lat: np.ndarray, hrrr_lon: np.ndarray) -> cKDTree:
    """Build a KD-tree from HRRR lat/lon arrays (converted to -180/180)."""
    lon180 = np.where(hrrr_lon > 180, hrrr_lon - 360, hrrr_lon)
    return cKDTree(np.column_stack([hrrr_lat.ravel(), lon180.ravel()]))


def resample_with_tree(
    tree: cKDTree,
    flat_values: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resample using a pre-built KD-tree.

    Parameters
    ----------
    tree : cKDTree
        Built from HRRR lat/lon.
    flat_values : np.ndarray
        Flattened HRRR field values (same order as tree input).
    target_lat, target_lon : np.ndarray
        2-D arrays of target grid lat/lon (degrees, -180/180 convention).

    Returns:
    -------
    np.ndarray
        Resampled field on the target grid shape.
    """
    pts = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    _, idx = tree.query(pts)
    return flat_values[idx].reshape(target_lat.shape)


# ---------------------------------------------------------------------------
# Lapse-rate correction
# ---------------------------------------------------------------------------


def lapse_rate_correct(
    t2m_celsius: np.ndarray,
    dem: np.ndarray,
    smooth_pixels: int = 100,
) -> np.ndarray:
    """Apply lapse-rate correction to adjust HRRR temperature to DEM elevation.

    HRRR reports temperature at its own (~3 km smoothed) terrain.  We
    approximate that by smoothing the high-res DEM to ~3 km (100 px at
    30 m) and correct:  ``T_adj = T_hrrr + LAPSE_RATE * (dem - dem_smooth)``

    Parameters
    ----------
    t2m_celsius : np.ndarray
        2-D temperature in °C on the target grid.
    dem : np.ndarray
        2-D DEM on the same grid (metres).
    smooth_pixels : int
        Smoothing kernel size (pixels) to approximate HRRR terrain.

    Returns:
    -------
    np.ndarray
        Lapse-rate adjusted temperature (°C, float32).
    """
    dem_clean = np.where(np.isfinite(dem), dem, 0.0)
    dem_smooth = uniform_filter(dem_clean.astype(np.float64), size=smooth_pixels)
    correction = LAPSE_RATE * (dem_clean - dem_smooth)
    return (t2m_celsius + correction).astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _projected_to_latlon(y: np.ndarray, x: np.ndarray, crs: CRS | str) -> tuple[np.ndarray, np.ndarray]:
    """Convert projected y/x 1-D coordinate arrays to 2-D lat/lon meshes."""
    crs_obj = CRS(crs)
    if crs_obj.is_geographic:
        # Already lat/lon — x is lon, y is lat
        lon2d, lat2d = np.meshgrid(x, y)
        return lat2d, lon2d

    transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(4326), always_xy=True)
    x2d, y2d = np.meshgrid(x, y)
    lon2d, lat2d = transformer.transform(x2d, y2d)
    return lat2d, lon2d


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def get_hrrr_for_dataset(
    ds: xr.Dataset,
    model: str | None = None,
    max_threads: int = 10,
) -> xr.DataArray:
    """Fetch nearest-overpass HRRR 2m temperature for every SAR time step.

    Uses ``FastHerbie`` for parallel batch downloads, then resamples
    each time step to the SAR grid and applies lapse-rate correction.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``time``, ``y``, ``x`` coordinates and a ``dem``
        variable.  Must have a CRS set (via ``rio.crs``).
    model : str or None
        ``"hrrrak"`` (Alaska, 3-hourly), ``"hrrr"`` (CONUS, hourly),
        or *None* to auto-detect from the dataset's bounding box.
    max_threads : int
        Number of parallel threads for FastHerbie downloads (default 10).

    Returns:
    -------
    xr.DataArray
        ``t2m`` with dims ``(time, y, x)`` in °C, lapse-rate adjusted.
        Times with no HRRR data are filled with NaN.
    """
    crs = ds.rio.crs
    if crs is None:
        raise ValueError('Dataset must have a CRS set (ds.rio.write_crs(...))')
    if 'dem' not in ds:
        raise ValueError("Dataset must contain a 'dem' variable for lapse-rate correction")

    if model is None:
        model = detect_hrrr_model(ds)

    times = pd.DatetimeIndex(ds.time.values)
    y_vals = ds.y.values
    x_vals = ds.x.values

    # Project grid to lat/lon for KD-tree queries
    target_lat, target_lon = _projected_to_latlon(y_vals, x_vals, crs)

    # Load DEM once (compute if dask-backed)
    dem = ds['dem'].values
    if dem.ndim == 3:
        dem = dem[0]
    dem = np.where(np.isfinite(dem), dem, 0.0)

    ny, nx = target_lat.shape
    t2m_out = np.full((len(times), ny, nx), np.nan, dtype=np.float32)

    # Build list of HRRR cycle datetimes to fetch, and a map back to SAR indices.
    # Multiple SAR timestamps may map to the same HRRR cycle (dedup).
    cycle_to_sar_indices: dict[str, list[int]] = {}
    for i, t in enumerate(times):
        cycle_hour = nearest_cycle_hour(t, model=model)
        cycle_dt = f'{t.strftime("%Y-%m-%d")} {cycle_hour:02d}:00'
        cycle_to_sar_indices.setdefault(cycle_dt, []).append(i)

    unique_cycles = sorted(cycle_to_sar_indices.keys())
    log.info(
        'Fetching HRRR %s t2m: %d SAR timestamps → %d unique cycles (max_threads=%d)',
        model,
        len(times),
        len(unique_cycles),
        max_threads,
    )

    # Batch download
    hrrr_ds = fetch_t2m_batch(unique_cycles, model=model, max_threads=max_threads)

    if hrrr_ds is None:
        log.warning('All HRRR downloads failed — returning NaN array')
        return xr.DataArray(
            t2m_out,
            dims=['time', 'y', 'x'],
            coords={'time': times, 'y': y_vals, 'x': x_vals},
            attrs=_make_attrs(model),
        )

    # Build KD-tree once from the HRRR grid
    hrrr_lat = hrrr_ds['latitude'].values  # (y, x)
    hrrr_lon = hrrr_ds['longitude'].values  # (y, x), 0–360
    tree = build_kdtree(hrrr_lat, hrrr_lon)

    # Pre-compute KD-tree query indices (same for all time steps)
    pts = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    _, kd_idx = tree.query(pts)

    # Resample + lapse-correct each HRRR time step
    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
    t2m_K_all = hrrr_ds['t2m'].values  # (n_cycles, hrrr_y, hrrr_x)

    for cycle_str in unique_cycles:
        cycle_ts = pd.Timestamp(cycle_str)
        # Find matching index in hrrr_ds (nearest, should be exact)
        hi = int(np.abs(hrrr_times - cycle_ts).argmin())

        # Resample using pre-computed indices
        flat_K = t2m_K_all[hi].ravel()
        t2m_sar = flat_K[kd_idx].reshape(ny, nx)
        t2m_C = t2m_sar + KELVIN_OFFSET
        t2m_adj = lapse_rate_correct(t2m_C, dem)

        # Write to all SAR time steps that map to this cycle
        for sar_idx in cycle_to_sar_indices[cycle_str]:
            t2m_out[sar_idx] = t2m_adj

    hrrr_ds.close()

    return xr.DataArray(
        t2m_out,
        dims=['time', 'y', 'x'],
        coords={'time': times, 'y': y_vals, 'x': x_vals},
        attrs=_make_attrs(model),
    )


def _make_attrs(model: str) -> dict:
    """Standard attributes for the t2m DataArray."""
    return {
        'units': 'degC',
        'long_name': '2m temperature (lapse-rate adjusted, nearest HRRR cycle)',
        'source': f'HRRR ({model}) via Herbie',
        'lapse_rate': f'{LAPSE_RATE} C/m',
    }
