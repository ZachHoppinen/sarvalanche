import xarray as xr
from scipy.stats import norm

from sarvalanche.utils.constants import eps

def pvalues_to_signed_z(p_channel: xr.DataArray, signs: xr.DataArray) -> xr.DataArray:
    """Convert p-values to signed z-scores."""
    # Clip to avoid numerical issues
    p_clip = xr.where(
        p_channel < eps, eps,
        xr.where(p_channel > 1 - eps, 1 - eps, p_channel)
    ).astype(float)

    # Convert to z-scores
    z = xr.apply_ufunc(
        norm.ppf,
        1.0 - p_clip,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Apply sign (increase vs decrease)
    z_signed = z * signs

    return z_signed