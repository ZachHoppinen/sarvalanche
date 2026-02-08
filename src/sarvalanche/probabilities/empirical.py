"""
Empirical backscatter change probability computation.

Computes probability based on magnitude of backscatter change across an avalanche date.
"""

import logging
import warnings
import xarray as xr

from sarvalanche.utils.validation import check_db_linear
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.weights.temporal import get_temporal_weights
from sarvalanche.weights.local_resolution import get_local_resolution_weights
from sarvalanche.weights.combinations import combine_weights, weighted_mean
from sarvalanche.probabilities.features import probability_backscatter_change

log = logging.getLogger(__name__)


def compute_track_empirical_probability(
    da: xr.DataArray,
    weights: xr.Dataset,
    avalanche_date,
    *,
    smooth_method: str | None = None,
    tau_days: float = 24.0,
    pair_dim: str = "pair",
) -> xr.DataArray:
    """
    Compute avalanche probability from backscatter change magnitude.

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
    tau_days : float, default=24.0
        Temporal decay scale for weighting observation pairs
    pair_dim : str, default="pair"
        Dimension name for observation pairs
    threshold_db : float, default=0.75
        Backscatter change (dB) where probability = 0.5
    logistic_slope : float, default=3.0
        Steepness of probability sigmoid

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

    # --- Backscatter changes crossing avalanche date ---
    diffs = backscatter_changes_crossing_date(da, avalanche_date, pair_dim=pair_dim)

    # --- Combine weights ---
    w_total = combine_weights(weights['w_temporal'], weights['w_resolution'])

    # --- Weighted mean change ---
    mean_change = weighted_mean(diffs, w_total, dim=pair_dim)

    # --- Convert to probability ---
    p = probability_backscatter_change(mean_change)

    return p