# """
# ECDF-based backscatter change probability computation.

# Computes probability based on statistical unusualness of post-event backscatter
# compared to pre-event distribution.
# """

# import logging
# import warnings
# import numpy as np
# import xarray as xr
# from scipy.stats import norm

# from sarvalanche.utils.validation import check_db_linear
# from sarvalanche.utils.constants import eps
# from sarvalanche.preprocessing.radiometric import linear_to_dB
# from sarvalanche.preprocessing.spatial import spatial_smooth
# from sarvalanche.features.stability import pixel_sigma_weighting
# from sarvalanche.features.incidence_angle import incidence_angle_weight
# from sarvalanche.features.weighting import combine_weights
# from sarvalanche.probabilities.combine import combine_z_to_probability

# log = logging.getLogger(__name__)

# import xarray as xr
# from scipy.stats import norm

# from sarvalanche.utils.constants import eps

# def combine_z_to_probability(
#     z_signed: xr.DataArray,
#     weights: xr.DataArray,
#     dim: str,
# ) -> xr.DataArray:
#     """Combine z-scores using Stouffer's method, convert to probability."""
#     valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)
#     w_eff = weights * valid

#     # Weighted Stouffer's method
#     num = (w_eff * z_signed).sum(dim, skipna=True)
#     den = np.sqrt((w_eff ** 2).sum(dim, skipna=True))
#     z_combined = num / (den + eps)

#     # Convert to probability
#     p = xr.apply_ufunc(
#         norm.cdf,
#         z_combined,
#         dask="parallelized",
#         output_dtypes=[float],
#     )

#     return p.clip(0, 1)

# def pvalues_to_signed_z(p_channel: xr.DataArray, signs: xr.DataArray) -> xr.DataArray:
#     """Convert p-values to signed z-scores."""
#     # Clip to avoid numerical issues
#     p_clip = xr.where(
#         p_channel < eps, eps,
#         xr.where(p_channel > 1 - eps, 1 - eps, p_channel)
#     ).astype(float)

#     # Convert to z-scores
#     z = xr.apply_ufunc(
#         norm.ppf,
#         1.0 - p_clip,
#         dask="parallelized",
#         output_dtypes=[float],
#     )

#     # Apply sign (increase vs decrease)
#     z_signed = z * signs

#     return z_signed

# def compute_track_ecdf_probability(
#     da: xr.DataArray,
#     lia: xr.DataArray,
#     avalanche_date,
#     *,
#     smooth_method: str | None = None,
#     min_ref: int = 4,
#     n_ref: int = 15,
#     lia_optimal: float = 55.0,
#     lia_width: float = 20.0,
# ) -> xr.DataArray:
#     """
#     Compute avalanche probability using ECDF-based statistical testing.

#     Parameters
#     ----------
#     da : xr.DataArray
#         Backscatter time series for single track/pol, dims=(time, y, x)
#     lia : xr.DataArray
#         Local incidence angle, dims=(y, x)
#     avalanche_date : datetime-like
#         Date separating pre/post observations
#     smooth_method : str, optional
#         Spatial smoothing method (None = no smoothing)
#     min_ref : int, default=4
#         Minimum number of pre-event observations required
#     n_ref : int, default=15
#         Maximum number of recent pre-event observations to use
#     lia_optimal : float, default=55.0
#         Optimal local incidence angle (degrees)
#     lia_width : float, default=20.0
#         Width of optimal angle range (degrees)

#     Returns
#     -------
#     xr.DataArray
#         Probability map, dims=(y, x)
#     """
#     # --- Convert to dB and smooth ---
#     if check_db_linear(da) != 'dB':
#         da = linear_to_dB(da)

#     if smooth_method is not None:
#         da = spatial_smooth(da, method=smooth_method)

#     # --- Split pre/post event ---
#     da_pre = da.sel(time=slice(None, avalanche_date))
#     da_post = da.sel(time=slice(avalanche_date, None))

#     if da_pre.time.size < min_ref:
#         log.warning(f"Insufficient pre-event observations: {da_pre.time.size} < {min_ref}")
#         return xr.full_like(da.isel(time=0), np.nan)

#     # Use most recent n_ref observations
#     da_ref = da_pre.isel(time=slice(-n_ref, None))

#     # --- Compute ECDF p-values ---
#     p_vals = _compute_ecdf_pvalues(da_ref, da_post, min_ref)  # (time, y, x)

#     # --- Stability weighting ---
#     sigma = da_ref.std()
#     w_stability = pixel_sigma_weighting(sigma)

#     # --- Incidence angle weighting ---
#     w_incidence = incidence_angle_weight(lia, lia_optimal, lia_width)

#     # --- Combine spatial weights ---
#     w_spatial = combine_weights(w_stability, w_incidence)  # (y, x)

#     # --- Sign of change ---
#     median = da_ref.median('time')
#     sign = xr.apply_ufunc(
#         np.sign,
#         da_post - median,
#         dask="parallelized",
#         output_dtypes=[np.int8],
#     )  # (time, y, x)

#     # --- Convert to signed z-scores ---
#     z_signed = pvalues_to_signed_z(p_vals, sign)  # (time, y, x)

#     # --- Combine across time using spatial weights ---
#     p = combine_z_to_probability(z_signed, w_spatial, dim='time')

#     return p


# def ecdf_survival_1pixel(ref_vals, post_vals, min_ref):
#     """P(reference <= post) - detects increases"""
#     ref_vals = ref_vals[~np.isnan(ref_vals)]
#     post_vals = np.atleast_1d(post_vals)
#     out = np.full(post_vals.shape, np.nan, dtype=np.float32)

#     if ref_vals.size < min_ref:
#         return out

#     for i, x in enumerate(post_vals):
#         if np.isnan(x):
#             continue
#         # number of reference pixels less than x
#         out[i] = np.mean(ref_vals >= x)

#     return out

# def _compute_ecdf_pvalues(
#     da_ref: xr.DataArray,
#     da_post: xr.DataArray,
#     min_ref: int,
# ) -> xr.DataArray:
#     """Compute empirical p-values via survival function."""

#     # Rename to avoid dimension name collision
#     da_ref = da_ref.rename({"time": "time_ref"})
#     da_post = da_post.rename({"time": "time_post"})

#     p = xr.apply_ufunc(
#         ecdf_survival_1pixel,
#         da_ref,
#         da_post,
#         kwargs={"min_ref": min_ref},
#         input_core_dims=[["time_ref"], ["time_post"]],
#         output_core_dims=[["time_post"]],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[np.float32],
#     )

#     # Rename back to standard "time" dimension
#     return p.rename({"time_post": "time"})
