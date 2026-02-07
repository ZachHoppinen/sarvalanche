"""
Probability combination utilities for fusing track/polarization results.
"""

import logging
import xarray as xr
import numpy as np
from typing import Literal

from scipy.stats import norm

from sarvalanche.utils.constants import eps

log = logging.getLogger(__name__)


def combine_probabilities(
    probs: xr.DataArray,
    dim: str,
    method: Literal['log_odds', 'stouffer', 'mean', 'product', 'max'] = 'log_odds',
    alpha: float | None = None,
    weights: xr.DataArray | None = None,
    eps: float = 1e-6,
) -> xr.DataArray:
    """
    Combine probabilities along a dimension.

    Parameters
    ----------
    probs : xr.DataArray
        Probability array to combine.
        Example dims: (source, y, x) or (track_pol, y, x)
    dim : str
        Dimension along which to combine (e.g., 'source', 'track_pol')
    method : str, default='log_odds'
        Combination method:
        - 'log_odds': Sum log-odds (Bayesian fusion)
        - 'stouffer': Weighted z-score combination
        - 'mean': Simple average
        - 'product': Geometric mean
        - 'max': Maximum probability
    alpha : float, optional
        Shrinkage toward 0.5 for log_odds (0=uniform, 1=full confidence)
    weights : xr.DataArray, optional
        Weights along the combination dimension.
        Must have `dim` in its dimensions and be broadcastable to probs.
        If None, uses equal weighting.
    eps : float, default=1e-6
        Numerical stability constant

    Returns
    -------
    xr.DataArray
        Combined probability with `dim` removed.

    Examples
    --------
    >>> # Combine list of results
    >>> results = [p1, p2, p3]  # Each (y, x)
    >>> probs = xr.concat(results, dim='source')  # (source, y, x)
    >>> p_combined = combine_temporal_probabilities(probs, dim='source')

    >>> # Combine with weights
    >>> weights = xr.DataArray([1.0, 0.8, 0.9], dims='source')
    >>> p_combined = combine_temporal_probabilities(probs, dim='source', weights=weights)
    """
    if dim not in probs.dims:
        raise ValueError(f"Dimension '{dim}' not found in probs. Available: {probs.dims}")

    log.info(f"Combining {probs.sizes[dim]} probabilities along '{dim}' using '{method}' method")

    # Apply shrinkage if requested
    if method == 'log_odds' and alpha is not None:
        log.debug(f"Applying shrinkage with alpha={alpha}")
        probs = 0.5 + alpha * (probs - 0.5)

    # Set up weights
    if weights is None:
        weights = xr.ones_like(probs.coords[dim], dtype=float)
    else:
        if dim not in weights.dims:
            raise ValueError(f"Weights must have dimension '{dim}'")

    # Combine
    if method == 'log_odds':
        combined = _combine_log_odds(probs, weights, dim, eps)
    elif method == 'stouffer':
        combined = _combine_stouffer(probs, weights, dim, eps)
    elif method == 'mean':
        combined = (probs * weights).sum(dim) / weights.sum(dim)
    elif method == 'product':
        log_vals = np.log(probs.clip(eps, 1 - eps))
        weighted_log = (log_vals * weights).sum(dim) / weights.sum(dim)
        combined = np.exp(weighted_log)
    elif method == 'max':
        combined = probs.max(dim)
    else:
        raise ValueError(f"Unknown method: {method}")

    return combined.clip(0, 1)


def _combine_log_odds(
    probs: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
    eps: float
) -> xr.DataArray:
    """Combine using weighted log-odds."""
    probs_clip = probs.clip(eps, 1 - eps)
    log_odds = np.log(probs_clip / (1 - probs_clip))
    log_odds_sum = (log_odds * weights).sum(dim, skipna=True)
    return 1.0 / (1.0 + np.exp(-log_odds_sum))


def _combine_stouffer(
    probs: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
    eps: float
) -> xr.DataArray:
    """Combine using weighted Stouffer's method."""
    from scipy.stats import norm

    probs_clip = probs.clip(eps, 1 - eps)
    z_scores = xr.apply_ufunc(
        norm.ppf,
        probs_clip,
        dask="parallelized",
        output_dtypes=[float]
    )

    num = (z_scores * weights).sum(dim, skipna=True)
    den = np.sqrt((weights ** 2).sum(dim))
    z_combined = num / (den + eps)

    return xr.apply_ufunc(
        norm.cdf,
        z_combined,
        dask="parallelized",
        output_dtypes=[float]
    )