import logging
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

def get_temporal_weights(
    times_1: xr.DataArray,
    times_2: xr.DataArray | np.datetime64 | pd.Timestamp,
    tau_days: float = 24,
) -> xr.DataArray:
    """
    Calculate temporal weights that sum to 1.0 based on time separation.
    Shorter intervals get higher weight (exponential decay), normalized to sum to 1.0.

    Parameters
    ----------
    times_1 : xr.DataArray
        First set of times (datetime64). Typically extracted as da['time']
    times_2 : xr.DataArray | datetime-like
        Either:
        - xr.DataArray of same length as times_1 (element-wise pairing)
        - Single datetime (all times_1 compared to this reference)
    tau_days : float
        Decay constant in days. Larger = slower decay.
    validate : bool
        If True, validate that weights sum to 1.0

    Returns
    -------
    xr.DataArray
        Temporal weights with same dimension as times_1, normalized to sum to 1.0

    Examples
    --------
    >>> # Case 1: Pair-wise between two coordinate arrays
    >>> da1 = xr.DataArray([...], dims=['time'])
    >>> da2 = xr.DataArray([...], dims=['time'])
    >>> weights = get_temporal_weights(da1['time'], da2['time'], tau_days=30)
    >>> # weights.sum() == 1.0

    >>> # Case 2: All times relative to single reference
    >>> da = xr.DataArray([...], dims=['time'])
    >>> event_time = np.datetime64('2024-03-15')
    >>> weights = get_temporal_weights(da['time'], event_time, tau_days=30)
    >>> # weights.sum() == 1.0
    """

    log.debug("get_temporal_weights: n_times=%d, tau_days=%s", times_1.size, tau_days)

    # Handle the two cases
    if isinstance(times_2, xr.DataArray):
        # Case 1: Element-wise pairing between two DataArrays
        if times_1.size != times_2.size:
            raise ValueError(
                f"times_1 and times_2 must have same length. "
                f"Got {times_1.size} and {times_2.size}"
            )
        dt = times_2.values - times_1.values
    else:
        # Case 2: Single reference time
        # Convert to numpy datetime64 for consistency
        ref_time = pd.Timestamp(times_2).to_datetime64()
        dt = times_1.values - ref_time

    # Convert timedelta to days
    dt_seconds = dt.astype("timedelta64[s]").astype(float)
    dt_days = dt_seconds / (24 * 3600)

    # Calculate exponential decay weights (unnormalized)
    w_temporal_unnorm = np.exp(-np.abs(dt_days) / tau_days)

    # Normalize to sum to 1.0
    w_temporal = w_temporal_unnorm / np.sum(w_temporal_unnorm)

    # Get the dimension name from times_1
    dim_name = times_1.dims[0] if times_1.dims else 'time'

    result = xr.DataArray(
        w_temporal,
        dims=[dim_name],
        coords={dim_name: times_1.values},
        name="w_temporal",
    )

    log.debug(
        "get_temporal_weights: weight range [%.4f, %.4f]",
        float(result.min()),
        float(result.max()),
    )

    return result