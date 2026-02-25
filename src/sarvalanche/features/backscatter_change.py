
import logging
from itertools import combinations
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

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

    log.debug(
        "backscatter_changes_crossing_date: %d time steps, date=%s",
        len(times), t0,
    )

    diffs = []
    t_start = []
    t_end = []

    combos = list(combinations(range(len(times)), 2))
    log.debug("backscatter_changes_crossing_date: checking %d time pairs", len(combos))
    for i, j in combos:
        ti, tj = times[i], times[j]

        if not (ti <= t0 < tj):
            continue

        diff = da.isel({time_dim: j}) - da.isel({time_dim: i})
        diffs.append(diff)
        t_start.append(ti)
        t_end.append(tj)

    log.debug("backscatter_changes_crossing_date: %d pairs cross date %s", len(diffs), t0)
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