
import logging
import warnings
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

    # Warn if the tightest crossing pair spans an unusually large gap.
    # Normal S1 revisit is 12 days (single satellite), so the minimum
    # crossing-pair span should be ≤24 days.  Anything larger indicates
    # missing acquisitions.
    min_span_days = min(
        (te - ts).days for ts, te in zip(t_start, t_end)
    )
    MAX_EXPECTED_SPAN_DAYS = 24
    if min_span_days > MAX_EXPECTED_SPAN_DAYS:
        warnings.warn(
            f"Smallest crossing-pair span is {min_span_days} days "
            f"(expected ≤{MAX_EXPECTED_SPAN_DAYS}). This likely indicates "
            f"missing Sentinel-1 acquisitions around {t0:%Y-%m-%d}. "
            f"Backscatter change estimates will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    stacked = xr.concat(diffs, dim=pair_dim)

    stacked = stacked.assign_coords(
        {
            "t_start": (pair_dim, t_start),
            "t_end": (pair_dim, t_end),
        }
    )

    stacked.name = name

    return stacked


def backscatter_changes_all_pairs(
    da: xr.DataArray,
    max_span_days: int = 60,
    time_dim: str = "time",
    pair_dim: str = "pair",
    name: str = "delta_backscatter",
):
    """
    Generate all unique backscatter changes (ti -> tj) with span <= max_span_days.

    Unlike backscatter_changes_crossing_date, this does not filter by a
    reference timestamp — it returns every pair in the time series that
    satisfies the span constraint.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension.
    max_span_days : int
        Maximum number of days between ti and tj.
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
    da = da.drop_vars('platform') if 'platform' in da.coords else da

    times = pd.to_datetime(da[time_dim].values)

    diffs = []
    t_start = []
    t_end = []

    for i, j in combinations(range(len(times)), 2):
        ti, tj = times[i], times[j]
        span = (tj - ti).days
        if span > max_span_days or span < 1:
            continue

        diff = da.isel({time_dim: j}) - da.isel({time_dim: i})
        diffs.append(diff)
        t_start.append(ti)
        t_end.append(tj)

    log.debug("backscatter_changes_all_pairs: %d pairs (max span %dd)",
              len(diffs), max_span_days)

    if not diffs:
        raise ValueError(
            f"No time pairs with span <= {max_span_days} days found."
        )

    stacked = xr.concat(diffs, dim=pair_dim)
    stacked = stacked.assign_coords(
        {
            "t_start": (pair_dim, t_start),
            "t_end": (pair_dim, t_end),
        }
    )
    stacked.name = name

    return stacked