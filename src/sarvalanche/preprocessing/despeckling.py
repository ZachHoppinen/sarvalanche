import logging
import numpy as np
from skimage.restoration import denoise_tv_chambolle

log = logging.getLogger(__name__)

def denoise_sar_homomorphic(da, tv_weight=0.1):
    """
    Homomorphic TV denoising for SAR gamma0 backscatter.

    Converts to dB, applies TV denoising, converts back to linear.
    Handles NaN/invalid pixels by mean-filling during denoising.

    Parameters
    ----------
    da : xarray.DataArray or numpy.ndarray
        SAR backscatter in linear scale (gamma0)
    tv_weight : float
        TV denoising weight. Higher = more smoothing. Default 0.1.

    Returns
    -------
    denoised : same type as input
        Denoised backscatter in linear scale
    """
    # assert check_db_linear(da) == 'linear'

    # Extract numpy array if xarray
    is_xarray = hasattr(da, 'values')
    arr = da.values if is_xarray else da

    # Build valid pixel mask
    valid = np.isfinite(arr) & (arr > 0)
    valid_fraction = valid.sum() / valid.size
    log.debug("denoise_sar_homomorphic: valid pixel fraction=%.3f (%d/%d)",
              valid_fraction, valid.sum(), valid.size)

    # Mean fill invalid pixels
    arr_filled = np.where(valid, arr, np.nanmean(arr[valid]))

    # Convert to dB
    arr_db = 10 * np.log10(arr_filled)

    # TV denoise
    arr_db_denoised = denoise_tv_chambolle(arr_db, weight=tv_weight)

    # Restore NaNs
    arr_db_denoised[~valid] = np.nan

    # Convert back to linear
    arr_denoised = 10 ** (arr_db_denoised / 10)
    arr_denoised[~valid] = np.nan

    # Return in original format
    if is_xarray:
        return da.copy(data=arr_denoised)
    else:
        return arr_denoised