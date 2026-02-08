
import numpy as np
import xarray as xr

def get_temporal_weights(
    times_1: xr.DataArray,
    times_2: xr.DataArray | np.datetime64 | pd.Timestamp,
    tau_days: float,
) -> xr.DataArray:
    """
    Calculate temporal weights based on time separation.
    Shorter intervals get higher weight (exponential decay).

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

    Returns
    -------
    xr.DataArray
        Temporal weights with same dimension as times_1

    Examples
    --------
    >>> # Case 1: Pair-wise between two coordinate arrays
    >>> da1 = xr.DataArray([...], dims=['time'])
    >>> da2 = xr.DataArray([...], dims=['time'])
    >>> weights = temporal_weights(da1['time'], da2['time'], tau_days=30)

    >>> # Case 2: All times relative to single reference
    >>> da = xr.DataArray([...], dims=['time'])
    >>> event_time = np.datetime64('2024-03-15')
    >>> weights = temporal_weights(da['time'], event_time, tau_days=30)
    """

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

    # Calculate exponential decay weights
    w_temporal = np.exp(-np.abs(dt_days) / tau_days)

    # Get the dimension name from times_1
    dim_name = times_1.dims[0] if times_1.dims else 'time'

    return xr.DataArray(
        w_temporal,
        dims=[dim_name],
        coords={dim_name: times_1.values},
        name="w_temporal",
    )