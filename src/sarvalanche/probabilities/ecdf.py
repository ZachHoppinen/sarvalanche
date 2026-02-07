"""
ECDF-based backscatter change probability computation.

Computes probability based on statistical unusualness of post-event backscatter
compared to pre-event distribution.
"""

import logging
import warnings
import numpy as np
import xarray as xr
from scipy.stats import norm

from sarvalanche.utils.validation import check_db_linear
from sarvalanche.utils.constants import eps
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.features.weighting import combine_weights
from sarvalanche.probabilities.conversions import pvalues_to_signed_z
from sarvalanche.probabilities.combine import combine_z_to_probability

log = logging.getLogger(__name__)


def compute_track_ecdf_probability(
    da: xr.DataArray,
    lia: xr.DataArray,
    avalanche_date,
    *,
    smooth_method: str | None = None,
    min_ref: int = 4,
    n_ref: int = 15,
    lia_optimal: float = 55.0,
    lia_width: float = 20.0,
) -> xr.DataArray:
    """
    Compute avalanche probability using ECDF-based statistical testing.

    Parameters
    ----------
    da : xr.DataArray
        Backscatter time series for single track/pol, dims=(time, y, x)
    lia : xr.DataArray
        Local incidence angle, dims=(y, x)
    avalanche_date : datetime-like
        Date separating pre/post observations
    smooth_method : str, optional
        Spatial smoothing method (None = no smoothing)
    min_ref : int, default=4
        Minimum number of pre-event observations required
    n_ref : int, default=15
        Maximum number of recent pre-event observations to use
    lia_optimal : float, default=55.0
        Optimal local incidence angle (degrees)
    lia_width : float, default=20.0
        Width of optimal angle range (degrees)

    Returns
    -------
    xr.DataArray
        Probability map, dims=(y, x)
    """
    # --- Convert to dB and smooth ---
    if check_db_linear(da) != 'dB':
        da = linear_to_dB(da)

    if smooth_method is not None:
        da = spatial_smooth(da, method=smooth_method)

    # --- Split pre/post event ---
    da_pre = da.sel(time=slice(None, avalanche_date))
    da_post = da.sel(time=slice(avalanche_date, None))

    if da_pre.time.size < min_ref:
        log.warning(f"Insufficient pre-event observations: {da_pre.time.size} < {min_ref}")
        return xr.full_like(da.isel(time=0), np.nan)

    # Use most recent n_ref observations
    da_ref = da_pre.isel(time=slice(-n_ref, None))

    # --- Compute ECDF p-values ---
    p_vals = _compute_ecdf_pvalues(da_ref, da_post, min_ref)  # (time, y, x)

    # --- Stability weighting ---
    sigma = da_ref.std()
    w_stability = pixel_sigma_weighting(sigma)

    # --- Incidence angle weighting ---
    w_incidence = incidence_angle_weight(lia, lia_optimal, lia_width)

    # --- Combine spatial weights ---
    w_spatial = combine_weights(w_stability, w_incidence)  # (y, x)

    # --- Sign of change ---
    median = da_ref.median('time')
    sign = xr.apply_ufunc(
        np.sign,
        da_post - median,
        dask="parallelized",
        output_dtypes=[np.int8],
    )  # (time, y, x)

    # --- Convert to signed z-scores ---
    z_signed = pvalues_to_signed_z(p_vals, sign)  # (time, y, x)

    # --- Combine across time using spatial weights ---
    p = combine_z_to_probability(z_signed, w_spatial, dim='time')

    return p

def _compute_ecdf_pvalues(
    da_ref: xr.DataArray,
    da_post: xr.DataArray,
    min_ref: int,
) -> xr.DataArray:
    """Compute empirical p-values via survival function."""
    def ecdf_survival_1pixel(ref_vals, post_vals, min_ref):
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        out = np.full(post_vals.shape, np.nan, dtype=np.float32)

        if ref_vals.size < min_ref:
            return out

        for i, x in enumerate(post_vals):
            if np.isnan(x):
                continue
            out[i] = np.mean(ref_vals >= x)

        return out

    time_post_len = da_post.sizes["time"]

    p = xr.apply_ufunc(
        ecdf_survival_1pixel,
        da_ref.rename({"time": "time_ref"}),
        da_post.rename({"time": "time_post"}),
        kwargs={"min_ref": min_ref},
        input_core_dims=[["time_ref"], ["time_post"]],
        output_core_dims=[["time_post"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={'output_sizes': {'new_dim': time_post_len}}
    )

    return p.transpose("time_post", "y", "x").rename({'time_post': 'time'})