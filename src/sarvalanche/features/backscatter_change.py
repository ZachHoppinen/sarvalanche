
from itertools import combinations
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.features.stability import pixel_stability_weight
from sarvalanche.features.temporal import temporal_pair_weights
from sarvalanche.features.incidence_angle import incidence_angle_weight

def backscatter_changes_crossing_date(
    da: xr.DataArray,
    timestamp,
    time_dim: str = "time",
    pair_dim: str = "pair",
    name: str = "delta_backscatter",
):
    """
    Generate all unique backscatter changes (ti -> tj) that cross a given timestamp
    and return them as a single stacked DataArray.

    A pair crosses the timestamp if:
        ti <= timestamp < tj

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension.
    timestamp : str or pandas.Timestamp
        Reference timestamp.
    time_dim : str, optional
        Name of the time dimension.
    pair_dim : str, optional
        Name of the stacked pair dimension.
    name : str, optional
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Stacked DataArray with dimension `pair` and coordinates
        `t_start`, `t_end`.
    """

    # Drop platform since we will be combining 1A, 1B, ... if from same orbit
    da = da.drop_vars('platform') if 'platform' in da.coords else da

    t0 = pd.to_datetime(timestamp)
    times = pd.to_datetime(da[time_dim].values)

    diffs = []
    t_start = []
    t_end = []

    combos = list(combinations(range(len(times)), 2))
    for i, j in combos:
        ti, tj = times[i], times[j]

        if not (ti <= t0 < tj):
            continue

        diff = da.isel({time_dim: j}) - da.isel({time_dim: i})
        diffs.append(diff)
        t_start.append(ti)
        t_end.append(tj)

    if not diffs:
        raise ValueError("No time pairs cross the given timestamp.")

    stacked = xr.concat(diffs, dim=pair_dim)

    stacked = stacked.assign_coords(
        {
            "t_start": (pair_dim, t_start),
            "t_end": (pair_dim, t_end),
        }
    )

    stacked.name = name

    return stacked

def combine_weights(
    w_temporal: xr.DataArray,
    w_stability: xr.DataArray,
    w_incidence: xr.DataArray | float,
) -> xr.DataArray:
    """
    Combine temporal, stability, and incidence weights.
    """
    w_total = w_temporal * w_stability * w_incidence
    return w_total.fillna(0).rename("w_total")

def weighted_pair_mean(
    diffs: xr.DataArray,
    weights: xr.DataArray,
    pair_dim: str = "pair",
) -> xr.DataArray:
    """
    Compute weighted mean over pair dimension.
    """
    return diffs.weighted(weights).mean(pair_dim)

def backscatter_change_weighted_mean(
    diffs: xr.DataArray,
    da: xr.DataArray,
    local_incidence_angle: xr.DataArray | None = None,
    avalanche_date=None,
    tau_days: float = 5.0,
    tau_variability: float = 1.0,
    incidence_power: float = 1.0,
    pair_dim: str = "pair",
) -> xr.DataArray:
    """
    Compute temporally and spatially weighted mean backscatter change.
    """

    w_temporal = temporal_pair_weights(
        diffs,
        tau_days=tau_days,
        pair_dim=pair_dim,
    )

    w_stability = pixel_stability_weight(
        da,
        tau_variability=tau_variability,
        avalanche_date=avalanche_date,
    )

    w_incidence = incidence_angle_weight(
        local_incidence_angle,
        incidence_power=incidence_power,
    )

    w_total = combine_weights(
        w_temporal,
        w_stability,
        w_incidence,
    )

    return weighted_pair_mean(
        diffs,
        w_total,
        pair_dim=pair_dim,
    )

