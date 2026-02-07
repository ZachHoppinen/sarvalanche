"""
Probability combination utilities for fusing multiple probabilistic estimates.
"""

import numpy as np
import xarray as xr
from typing import Literal
from scipy.stats import norm
import logging

from sarvalanche.utils.constants import eps

log = logging.getLogger(__name__)


def combine_probabilities_weighted(
    probs: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
    method: Literal['log_odds', 'stouffer', 'mean', 'product'] = 'log_odds',
    eps: float = 1e-6,
) -> xr.DataArray:
    """
    Combine probabilities along a dimension using weighted aggregation.

    Supports multiple combination methods for fusing probabilistic estimates
    with spatially-varying or observation-specific weights.

    Parameters
    ----------
    probs : xr.DataArray
        Probabilities in [0, 1] to combine.
        Example dims: (obs, y, x) or (pair, y, x) or (time, y, x)
    weights : xr.DataArray
        Weights for each observation.
        Must be broadcastable to probs shape.
        Example dims: (obs, y, x) or (y, x) [broadcasts to obs]
    dim : str
        Dimension along which to combine (e.g., 'obs', 'pair', 'time').
    method : str, default='log_odds'
        Combination method:
        - 'log_odds': Sum weighted log-odds (Bayesian-ish)
        - 'stouffer': Weighted Stouffer's z-score method (statistical meta-analysis)
        - 'mean': Weighted arithmetic mean
        - 'product': Weighted geometric mean (product of probabilities)
    eps : float, default=1e-6
        Small constant for numerical stability.

    Returns
    -------
    xr.DataArray
        Combined probability in [0, 1], with `dim` removed.

    Examples
    --------
    >>> # Combine observation probabilities with pixel-based weights
    >>> p_obs = xr.DataArray(...)  # (obs, y, x)
    >>> w = xr.DataArray(...)      # (obs, y, x) or (y, x)
    >>> p_combined = combine_probabilities_weighted(p_obs, w, dim='obs', method='log_odds')

    >>> # Combine temporal pairs with temporal weights
    >>> p_pairs = xr.DataArray(...)  # (pair, y, x)
    >>> w_temporal = xr.DataArray(...)  # (pair, y, x)
    >>> p_combined = combine_probabilities_weighted(p_pairs, w_temporal, dim='pair')

    Notes
    -----
    **Method details:**

    log_odds:
        Converts p to log(p/(1-p)), sums with weights, converts back.
        Treats probabilities as independent evidence.

    stouffer:
        Converts p to z-scores via norm.ppf(), weighted sum, converts back.
        Standard method for combining p-values from statistical tests.

    mean:
        Simple weighted average. Not statistically principled but interpretable.

    product:
        Geometric mean: prod(p^w)^(1/sum(w)).
        Assumes independence, conservative (reduces probability).
    """
    # Validate inputs
    if dim not in probs.dims:
        raise ValueError(f"Dimension '{dim}' not found in probs. Available: {probs.dims}")

    # Handle NaN observations
    valid = xr.where(np.isfinite(probs), 1.0, 0.0)
    w_eff = weights * valid

    # Clip probabilities to valid range
    probs_clip = probs.clip(eps, 1 - eps)

    if method == 'log_odds':
        # Convert to log-odds
        log_odds = np.log(probs_clip / (1 - probs_clip))

        # Weighted sum
        log_odds_sum = (w_eff * log_odds).sum(dim, skipna=True)

        # Convert back to probability
        p_combined = 1.0 / (1.0 + np.exp(-log_odds_sum))

    elif method == 'stouffer':
        # Convert to z-scores (assuming probs are p-values from tests)
        z_scores = xr.apply_ufunc(
            norm.ppf,
            probs_clip,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Weighted Stouffer's method
        num = (w_eff * z_scores).sum(dim, skipna=True)
        den = np.sqrt((w_eff ** 2).sum(dim, skipna=True))
        z_combined = num / (den + eps)

        # Convert back to probability
        p_combined = xr.apply_ufunc(
            norm.cdf,
            z_combined,
            dask="parallelized",
            output_dtypes=[float],
        )

    elif method == 'mean':
        # Weighted arithmetic mean
        num = (w_eff * probs).sum(dim, skipna=True)
        den = w_eff.sum(dim, skipna=True)
        p_combined = num / (den + eps)

    elif method == 'product':
        # Weighted geometric mean
        log_probs = np.log(probs_clip)
        weighted_log_sum = (w_eff * log_probs).sum(dim, skipna=True)
        weight_sum = w_eff.sum(dim, skipna=True)
        p_combined = np.exp(weighted_log_sum / (weight_sum + eps))

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: log_odds, stouffer, mean, product")

    # Ensure output is in [0, 1]
    p_combined = p_combined.clip(0, 1)

    log.debug(f"Combined probabilities along '{dim}' using '{method}' method")

    return p_combined


def combine_signed_pvalues_weighted(
    p_vals: xr.DataArray,
    signs: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    """
    Combine directional p-values using weighted Stouffer's method.

    Designed for ECDF-style analysis where p-values indicate "unusualness"
    and signs indicate direction of change.

    Parameters
    ----------
    p_vals : xr.DataArray
        P-values in [0, 1], dims including `dim`.
    signs : xr.DataArray
        Signs {-1, 0, 1} indicating direction, same dims as p_vals.
    weights : xr.DataArray
        Weights, broadcastable to p_vals.
    dim : str
        Dimension to combine along.
    eps : float
        Numerical stability constant.

    Returns
    -------
    xr.DataArray
        Combined probability of change in [0, 1].

    Examples
    --------
    >>> p_vals = compute_ecdf_pvalues(...)  # (time, y, x)
    >>> signs = compute_signs(...)          # (time, y, x)
    >>> weights = compute_weights(...)      # (y, x)
    >>> p_combined = combine_signed_pvalues_weighted(p_vals, signs, weights, dim='time')
    """
    # Clip p-values
    p_clip = p_vals.clip(eps, 1 - eps)

    # Convert to z-scores
    z_scores = xr.apply_ufunc(
        norm.ppf,
        1.0 - p_clip,  # Survival function convention
        dask="parallelized",
        output_dtypes=[float],
    )

    # Apply signs
    z_signed = z_scores * signs

    # Handle NaN
    valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)
    w_eff = weights * valid

    # Weighted Stouffer's method
    num = (w_eff * z_signed).sum(dim, skipna=True)
    den = np.sqrt((w_eff ** 2).sum(dim, skipna=True))
    z_combined = num / (den + eps)

    # Convert to probability (two-tailed)
    p_combined = xr.apply_ufunc(
        norm.cdf,
        z_combined,
        dask="parallelized",
        output_dtypes=[float],
    )

    return p_combined