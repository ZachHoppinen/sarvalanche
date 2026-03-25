"""Build static terrain stack for pairwise debris detector."""

import logging

import numpy as np
import xarray as xr

from sarvalanche.ml.pairwise_debris_classifier.channels import (
    STATIC_CHANNELS,
    normalize_static_channel,
)
from sarvalanche.utils.validation import check_rad_degrees

log = logging.getLogger(__name__)


def _estimate_utm_crs(ds):
    """Estimate UTM CRS for a dataset from its bounding box."""
    from shapely.geometry import box
    from sarvalanche.utils.projections import find_utm_crs

    bounds = ds.rio.bounds()
    aoi = box(*bounds)
    crs = ds.rio.crs
    return find_utm_crs(aoi, crs)


def build_static_stack(ds: xr.Dataset) -> np.ndarray:
    """Build (len(STATIC_CHANNELS), H, W) static terrain stack.

    Channels: slope, aspect_northing, aspect_easting, cell_counts, tpi.
    All channels are globally normalized via normalize_static_channel.

    Aspect must be in the dataset as radians or degrees (auto-detected
    via check_rad_degrees and converted if needed).

    TPI is computed from the DEM. If the dataset is in a geographic CRS,
    the DEM is temporarily reprojected to the appropriate UTM zone for
    TPI computation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with static variables (slope, aspect, dem, cell_counts).
        Must have a CRS set via rioxarray.
    """
    H, W = ds.sizes['y'], ds.sizes['x']
    n_channels = len(STATIC_CHANNELS)
    stack = np.zeros((n_channels, H, W), dtype=np.float32)

    # ── Aspect decomposition (only if needed by STATIC_CHANNELS) ─────
    aspect_derived = {}
    need_aspect = {'aspect_northing', 'aspect_easting'} & set(STATIC_CHANNELS)
    if need_aspect and 'aspect' in ds.data_vars:
        aspect_da = ds['aspect']
        unit = check_rad_degrees(aspect_da)
        aspect_arr = np.nan_to_num(aspect_da.values.astype(np.float32), nan=0.0)
        if unit == 'degrees':
            log.info('  Aspect is in degrees, converting to radians')
            aspect_arr = np.deg2rad(aspect_arr)
        aspect_derived['aspect_northing'] = np.cos(aspect_arr)
        aspect_derived['aspect_easting'] = np.sin(aspect_arr)
    elif need_aspect:
        log.warning('aspect_northing/easting requested but aspect not in dataset — will be zeros')

    # ── TPI from DEM (only if needed by STATIC_CHANNELS) ────────────
    derived = {}
    if 'tpi' in STATIC_CHANNELS and 'dem' in ds.data_vars:
        from sarvalanche.utils.terrain import compute_tpi

        dem_da = ds['dem']
        try:
            if ds.rio.crs and ds.rio.crs.is_geographic:
                utm_crs = _estimate_utm_crs(ds)
                log.info('  Reprojecting DEM to %s for TPI computation', utm_crs)
                dem_proj = dem_da.rio.reproject(utm_crs)
                tpi_da = compute_tpi(dem_proj, radius_m=300.0)
                tpi_da = tpi_da.rio.reproject_match(dem_da)
                tpi_arr = tpi_da.values.astype(np.float32)
                if tpi_arr.shape != (H, W):
                    raise ValueError(
                        f"TPI shape {tpi_arr.shape} doesn't match dataset ({H}, {W}) "
                        f"after reproject_match — projection mismatch")
                derived['tpi'] = np.nan_to_num(tpi_arr, nan=0.0)
            else:
                tpi_da = compute_tpi(dem_da, radius_m=300.0)
                derived['tpi'] = np.nan_to_num(tpi_da.values.astype(np.float32), nan=0.0)
            log.info('  Computed TPI from DEM')
        except Exception:
            log.warning('Could not compute TPI', exc_info=True)
            derived['tpi'] = np.zeros((H, W), dtype=np.float32)
    elif 'tpi' in STATIC_CHANNELS:
        log.warning('tpi requested but dem not in dataset — will be zeros')

    # ── Fill static stack ────────────────────────────────────────────
    for ch, var in enumerate(STATIC_CHANNELS):
        if var in aspect_derived:
            stack[ch] = aspect_derived[var]
        elif var in derived:
            stack[ch] = normalize_static_channel(derived[var], var)
        elif var in ds.data_vars:
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            stack[ch] = normalize_static_channel(arr, var)
        else:
            log.warning('Static channel %r not found in dataset or derived — will be zeros', var)

    return stack
