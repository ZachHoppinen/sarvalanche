
import logging
import warnings
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

def linear_to_dB(da: xr.DataArray):
    input_units = da.attrs.get('units', 'unknown')
    if da.attrs.get('units', '').lower() == 'db':
        log.debug("linear_to_dB: already-dB guard triggered (units=%s), returning unchanged", input_units)
        return da
    log.debug("linear_to_dB: converting from units=%s", input_units)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero', RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)

        # This will create -inf for zeros and NaN for negatives
        da_db = 10 * np.log10(da)

        # Replace -inf and invalid values with NaN
        da_db = xr.where(np.isfinite(da_db), da_db, np.nan)
    da_db.attrs['units'] = 'db'

    return da_db

def dB_to_linear(da: xr.DataArray):
    if da.attrs['units'].lower() == 'linear':
        log.debug("dB_to_linear: already-linear guard triggered, returning unchanged")
        return da

    da = 10**(da / 10)
    da.attrs['units'] = 'linear'
    return da

def normalize_to_stable_areas(da: xr.DataArray, stable_mask: xr.DataArray) -> xr.DataArray:
    """
    Normalize each time slice to median of stable areas.

    Parameters
    ----------
    da : xr.DataArray
        Backscatter in dB, dims=(time, y, x)
    stable_mask : xr.DataArray
        Boolean mask of stable areas (e.g., flat terrain, rock), dims=(y, x)
    """
    # Compute median offset for each time slice
    stable_pixel_count = int(stable_mask.sum())
    log.debug("normalize_to_stable_areas: stable pixel count=%d", stable_pixel_count)

    stable_median = da.where(stable_mask).median(dim=['y', 'x'])

    # Reference: overall median of stable areas
    reference_median = stable_median.median(dim='time')

    # Normalize each time slice
    offset = stable_median - reference_median
    log.debug(
        "normalize_to_stable_areas: offset range [%.4f, %.4f]",
        float(offset.min()),
        float(offset.max()),
    )
    da_normalized = da - offset

    return da_normalized