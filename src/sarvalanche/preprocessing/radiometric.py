import numpy as np
import xarray as xr

def linear_to_dB(da: xr.DataArray):
    if da.attrs['units'].lower() == 'db':
        return da

    da = 10 * np.log10(da)
    da.attrs['units'] = 'db'

    return da

def dB_to_linear(da: xr.DataArray):
    if da.attrs['units'].lower() == 'linear':
        return da

    da = 10**(da / 10)
    da.attrs['units'] = 'linear'
    return da