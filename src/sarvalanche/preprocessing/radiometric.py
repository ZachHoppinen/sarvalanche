
import warnings
import numpy as np
import xarray as xr

def linear_to_dB(da: xr.DataArray):
    if da.attrs.get('units', '').lower() == 'db':
        return da

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
    stable_median = da.where(stable_mask).median(dim=['y', 'x'])

    # Reference: overall median of stable areas
    reference_median = stable_median.median(dim='time')

    # Normalize each time slice
    offset = stable_median - reference_median
    da_normalized = da - offset

    return da_normalized