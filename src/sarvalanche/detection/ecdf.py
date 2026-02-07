"""
ECDF-based backscatter change detection for avalanche debris.

This module implements avalanche detection based on statistical comparison of
post-event backscatter against the pre-event distribution. For each track/polarization:
1. Convert to dB and optionally smooth
2. Compute ECDF p-values (how unusual is post-event backscatter?)
3. Convert to z-scores with directional sign
4. Weight by pixel stability, viewing geometry, and polarization quality
5. Combine weighted z-scores and convert to probability
6. Fuse across tracks/pols
"""

import logging
import warnings
import numpy as np
import xarray as xr
from scipy.stats import norm

from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.utils.constants import eps
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.weighting import combine_weights
from sarvalanche.detection.probability import log_odds_combine, _z_to_probability

log = logging.getLogger(__name__)


def calculate_ecdf_backscatter_probability(
    ds: xr.Dataset,
    avalanche_date,
    *,
    polarizations=("VV", "VH"),
    smooth_method=None,
    min_ref: int = 4,
    n_ref: int = 15,
    q_pol: dict = {'VV': 1.0, 'VH': 0.8},
    lia_optimal: float = 55.0,
    lia_width: float = 20.0,
    beta: float = 10.0,
    z_pivot: float = 0.5,
    combine_alpha: float = 0.5,
):
    """
    Compute avalanche debris probability using ECDF-based statistical testing.

    [docstring same as before...]
    """
    log.info(f"Computing ECDF probabilities for {len(polarizations)} polarizations")

    probability_list: list[xr.DataArray] = []

    for track, pol, da, lia in iter_track_pol_combinations(
        ds,
        polarizations=polarizations,
        include_lia=True,
        skip_missing=True
    ):
        log.debug(f"Processing track={track}, pol={pol}")

        # --- Convert to dB and smooth ---
        if check_db_linear(da) != 'dB':
            da = linear_to_dB(da)

        if smooth_method is not None:
            da = spatial_smooth(da, method=smooth_method)

        # --- Split pre/post event ---
        da_pre = da.sel(time=slice(None, avalanche_date))
        da_post = da.sel(time=slice(avalanche_date, None))

        if da_pre.time.size < min_ref:
            log.warning(f"Skipping {pol}_{track}: only {da_pre.time.size} pre-event observations")
            continue

        # Use most recent n_ref observations as reference
        da_ref = da_pre.isel(time=slice(-n_ref, None))

        # --- Compute ECDF p-values for each post-event observation ---
        p_vals = _compute_ecdf_pvalues(da_ref, da_post, min_ref)  # (time, y, x)

        # --- Pre-event stability (sigma) ---
        sigma = da_ref.std('time')  # (y, x)
        w_stability = pixel_sigma_weighting(sigma)

        # --- Sign of change (increase vs decrease) ---
        median = da_ref.median('time')
        sign = xr.apply_ufunc(
            np.sign,
            da_post - median,
            dask="parallelized",
            output_dtypes=[np.int8],
        )  # (time, y, x)

        # --- Convert p-values to signed z-scores ---
        z_signed = _pvalues_to_signed_z(p_vals, sign)  # (time, y, x)

        # --- Incidence angle weights ---
        w_incidence = incidence_angle_weight(lia, lia_optimal, lia_width)  # (y, x)

        # --- Polarization weight ---
        w_pol = q_pol.get(pol, 0.5)  # scalar

        # --- Combine weights (same as empirical!) ---
        w_total = combine_weights(
            w_stability,  # (y, x)
            w_incidence,  # (y, x)
            w_pol,        # scalar
        )  # (y, x)

        # --- Weighted combination of z-scores across time ---
        z_combined = _combine_z_across_time(z_signed, w_total)  # (y, x)

        # --- Convert to probability ---
        p_single = _z_to_probability(z_combined, beta=beta, z_pivot=z_pivot)
        probability_list.append(p_single)

    if not probability_list:
        raise ValueError(
            "No valid track/polarization combinations found. "
            "Check data availability and min_ref threshold."
        )

    # --- Combine across tracks/pols (same as empirical!) ---
    log.info(f"Combining {len(probability_list)} probability maps with alpha={combine_alpha}")

    probability_combined = log_odds_combine(
        probability_list,
        alpha=combine_alpha,
    )

    return probability_combined


def _combine_z_across_time(z_signed: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """
    Combine z-scores across time dimension using spatial weights.

    Parameters
    ----------
    z_signed : xr.DataArray
        Signed z-scores, dims=(time, y, x)
    weights : xr.DataArray
        Spatial weights, dims=(y, x)

    Returns
    -------
    xr.DataArray
        Combined z-score, dims=(y, x)
    """
    # Each time observation gets the same spatial weight pattern
    # We're averaging z-scores across time, weighted by pixel reliability

    valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)

    # Broadcast weights to time dimension
    w_broadcast = weights  # xarray will broadcast automatically

    w_eff = w_broadcast * valid

    # Weighted mean across time
    num = (w_eff * z_signed).sum("time", skipna=True)
    den = w_eff.sum("time", skipna=True)

    z_mean = num / (den + 1e-6)

    return z_mean

def _compute_ecdf_pvalues(
    da_ref: xr.DataArray,
    da_post: xr.DataArray,
    min_ref: int,
) -> xr.DataArray:
    """
    Compute empirical p-values via survival function.

    For each post-event observation, compute P(pre-event >= post-event).
    """
    def ecdf_survival_1pixel(ref_vals, post_vals, min_ref):
        """Compute survival probabilities for a single pixel."""
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

    p = p.transpose("time_post", "y", "x").rename({'time_post': 'time'})
    return p

def _pvalues_to_signed_z(p_channel: xr.DataArray, signs: xr.DataArray) -> xr.DataArray:
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