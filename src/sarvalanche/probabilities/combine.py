"""
Probability combination utilities for fusing track/polarization results.
"""

import logging
import xarray as xr
import numpy as np
from typing import Literal

log = logging.getLogger(__name__)

def combine_probabilities(
    probs: xr.DataArray,
    weights: xr.DataArray | None = None,
    dim: str = 'track_pol',
    method: str = 'weighted_mean',
    agreement_boosting: bool = False,
    min_prob_threshold: float = 0.1,
    agreement_strength: float = 0.8,
) -> xr.DataArray:
    """
    Combine probabilities using weighted mean or log-odds.

    Parameters
    ----------
    probs : xr.DataArray
        Probabilities to combine (values between 0 and 1)
    weights : xr.DataArray, optional
        Weights for each probability (should sum to 1.0 along dim)
    dim : str
        Dimension to combine along
    method : str
        'weighted_mean' or 'log_odds'
    agreement_boosting : bool
        If True, boost probability when multiple sources agree
    min_prob_threshold : float
        Minimum probability to consider as "detection" for agreement boosting
    agreement_strength : float
        Boost strength for agreement (0.0-1.0)

    Returns
    -------
    xr.DataArray
        Combined probability (0-1)

    Examples
    --------
    >>> # Simple weighted average
    >>> p = combine_probabilities(probs, weights, method='weighted_mean')

    >>> # Log-odds combination (better for extreme probabilities)
    >>> p = combine_probabilities(probs, weights, method='log_odds')

    >>> # With agreement boosting
    >>> p = combine_probabilities(
    ...     probs, weights,
    ...     agreement_boosting=True,
    ...     agreement_strength=0.8
    ... )
    """

    # Validate method
    if method not in ['weighted_mean', 'log_odds']:
        raise ValueError(
            f"Unknown method: {method}. Use 'weighted_mean' or 'log_odds'."
        )

    # Set default weights if not provided
    if weights is None:
        n = probs.sizes[dim]
        weights = xr.ones_like(probs) / n

    # Combine probabilities based on method
    if method == 'weighted_mean':
        # Standard weighted average
        p_combined = (probs * weights).sum(dim=dim)

    elif method == 'log_odds':
        # Log-odds space combination
        # log_odds = log(p / (1-p))
        # Better for probabilities near 0 or 1

        epsilon = 1e-10  # Avoid division by zero
        probs_safe = probs.clip(epsilon, 1 - epsilon)

        # Convert to log-odds
        log_odds = np.log(probs_safe / (1 - probs_safe))

        # Weighted average in log-odds space
        log_odds_combined = (log_odds * weights).sum(dim=dim)

        # Convert back to probability
        # p = 1 / (1 + exp(-log_odds))
        p_combined = 1 / (1 + np.exp(-log_odds_combined))

    # Apply agreement boosting if enabled
    if agreement_boosting:
        # Filter out low probabilities
        probs_filtered = probs.where(probs >= min_prob_threshold)

        # Count valid sources (how many detect something)
        n_valid = probs_filtered.notnull().sum(dim=dim)
        n_total = probs.sizes[dim]

        # Agreement: fraction showing detection
        agreement = n_valid / n_total

        # Boost formula: p_final = p_mean + (1 - p_mean) × agreement² × strength
        boost = (1 - p_combined) * (agreement ** 2) * agreement_strength
        p_combined = p_combined + boost

        # Handle no valid sources
        p_combined = xr.where(n_valid > 0, p_combined, 0.0)

    return p_combined.clip(0, 1)