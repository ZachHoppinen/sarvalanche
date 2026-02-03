
import numpy as np
import xarray as xr

def pixel_stability_weight(
    da: xr.DataArray,
    tau_variability: float,
    avalanche_date=None,
) -> xr.DataArray:
    """
    Weight pixels by temporal stability using MAD.
    More variable pixels receive lower weight.
    """
    da_stable = da
    if avalanche_date is not None:
        da_stable = da.sel(time=da.time < avalanche_date)

    median_pixel = da_stable.median(dim="time")
    mad_pixel = np.abs(da_stable - median_pixel).median(dim="time")

    w_stability = np.exp(-mad_pixel / tau_variability)

    return w_stability.fillna(0).rename("w_stability")
