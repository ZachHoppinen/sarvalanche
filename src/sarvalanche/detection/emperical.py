"""
Empirical backscatter change detection for avalanche debris.

This module implements avalanche detection based on direct measurement of
backscatter changes across an avalanche date. For each track/polarization:
1. Convert to dB and optionally smooth
2. Compute temporal differences crossing the avalanche date
3. Weight by temporal proximity, pixel stability, and viewing geometry
4. Convert weighted change to probability
5. Combine across tracks/pols using log-odds fusion
"""

import logging
import xarray as xr

from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.temporal import temporal_pair_weights
from sarvalanche.features.weighting import combine_weights, weighted_mean
from sarvalanche.detection.probability import log_odds_combine, probability_backscatter_change

log = logging.getLogger(__name__)


def calculate_empirical_backscatter_probability(
    ds: xr.Dataset,
    avalanche_date,
    *,
    polarizations=("VV", "VH"),
    smooth_method=None,
    tau_days=24,
    combine_alpha=0.5,
    pair_dim: str = "pair",
):
    """
    Compute avalanche debris probability from SAR backscatter changes.

    For each (track, polarization) pair:
      1. Convert backscatter to dB
      2. Spatially smooth (optional)
      3. Compute pre/post-avalanche backscatter changes
      4. Aggregate changes using temporal, stability, and incidence weighting
      5. Convert to probability

    All per-track/pol probabilities are then fused using log-odds
    combination with optional shrinkage toward 0.5.

    Parameters
    ----------
    ds : xr.Dataset
        Canonical SAR dataset containing:
        - VV, VH backscatter (time, y, x)
        - track (time)
        - lia (static_track, y, x)
    avalanche_date : str or datetime-like
        Date separating pre/post-event acquisitions.
    polarizations : tuple[str], default=("VV", "VH")
        Polarizations to include.
    smooth_method : str, optional
        Spatial smoothing method passed to `spatial_smooth`.
        None = no smoothing.
    tau_days : float, default=24
        Temporal decay scale (days) for weighting observation pairs.
        Pairs closer to avalanche_date receive higher weight.
    combine_alpha : float, default=0.5
        Shrinkage factor toward 0.5 when combining probabilities.
        0 = uniform prior, 1 = full confidence.
    pair_dim : str, default="pair"
        Name of the dimension along which observation pairs are stacked.

    Returns
    -------
    xr.DataArray
        Combined backscatter-change probability in [0, 1], dims=(y, x).

    Raises
    ------
    ValueError
        If no valid track/polarization combinations produce probabilities.

    Examples
    --------
    >>> ds = load_sar_dataset(...)
    >>> p_debris = calculate_empirical_backscatter_probability(
    ...     ds,
    ...     avalanche_date="2020-01-15",
    ...     tau_days=30,
    ...     combine_alpha=0.7
    ... )
    """
    log.info(f"Computing empirical backscatter probabilities for {len(polarizations)} "
             f"polarizations with tau_days={tau_days}")

    p_delta_list: list[xr.DataArray] = []

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

        # --- Backscatter change across avalanche date ---
        diffs = backscatter_changes_crossing_date(da, avalanche_date)

        # --- temporal weighting ---
        w_temporal = temporal_pair_weights(
            diffs,
            tau_days=tau_days,
            pair_dim=pair_dim,
        )

        # --- radiometric stability weighting ---
        sigma_db = da.sel(time=slice(None, avalanche_date)).std(dim='time')
        w_stability = pixel_sigma_weighting(sigma_db)

        # --- Incidence angle weighting ---
        w_incidence = incidence_angle_weight(lia)

        # --- Combine weights ---
        w_total = combine_weights(
            w_temporal,
            w_stability,
            w_incidence,
        )

        # --- Weighted mean ---
        mean_change = weighted_mean(
            diffs,
            w_total,
            dim="pair",
        )

        # --- 4. Probability mapping ---
        p_delta = probability_backscatter_change(mean_change)
        p_delta_list.append(p_delta)

    if not p_delta_list:
        raise ValueError(
            "No backscatter probabilities were computed. "
            "Check that dataset contains requested polarizations and valid data."
        )

    log.info(f"Combining {len(p_delta_list)} probability maps with alpha={combine_alpha}")

    # --- 5. Combine across tracks / pols ---
    p_delta_combined = log_odds_combine(
        p_delta_list,
        alpha=combine_alpha,
    )

    return p_delta_combined