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
from sarvalanche.weights.combinations import combine_weights, weighted_mean
from sarvalanche.probabilities.features import probability_backscatter_change

from sarvalanche.utils.validation import validate_weights_sum_to_one

log = logging.getLogger(__name__)


def compute_track_empirical_probability(
    da: xr.DataArray,
    # weights: xr.Dataset,
    avalanche_date,
    *,
    smooth_method: str | None = None,
    pair_dim: str = "pair",
) -> xr.DataArray:
    """
    Compute avalanche probability from backscatter change magnitude.

    Parameters
    ----------
    da : xr.DataArray
        Backscatter time series for single track/pol, dims=(time, y, x)
    avalanche_date : datetime-like
        Date separating pre/post observations
    smooth_method : str, optional
        Spatial smoothing method (None = no smoothing)
    tau_days : float, default=24.0
        Temporal decay scale for weighting observation pairs
    pair_dim : str, default="pair"
        Dimension name for observation pairs

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(probability, mean_change)`` — the probability map and the
        weighted-mean backscatter change in dB, both with dims ``(y, x)``.
    """
    # --- Convert to dB and smooth ---
    if check_db_linear(da) != 'dB':
        log.info('Converting to dB')
        da = linear_to_dB(da)

    if smooth_method is not None:
        da = spatial_smooth(da, method=smooth_method)

    # --- Backscatter changes crossing avalanche date ---
    diffs = backscatter_changes_crossing_date(da, avalanche_date, pair_dim=pair_dim)
    w_pair_temporal = get_temporal_weights(diffs['t_start'], diffs['t_end'])
    validate_weights_sum_to_one(w_pair_temporal, dim = 'pair')

    # --- Combine weights ---
    # w_total = combine_weights(w_pair_temporal, weights['w_resolution'])

    # --- Weighted mean change ---
    mean_change = weighted_mean(diffs, w_pair_temporal, dim=pair_dim)

    # --- Convert to probability ---
    p = probability_backscatter_change(mean_change)

    return p, mean_change