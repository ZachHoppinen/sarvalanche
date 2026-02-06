
from itertools import combinations
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.temporal import temporal_pair_weights
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.preprocessing.radiometric import linear_to_dB

from sarvalanche.utils.validation import check_db_linear, check_rad_degrees

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