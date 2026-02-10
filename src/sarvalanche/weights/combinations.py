
from functools import reduce
import operator
import logging

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def combine_weights(
    *weights: xr.DataArray | float,
    dim: str | None = None,
) -> xr.DataArray:
    """
    Combine multiple weights through multiplication and normalize to sum to 1.0.

    Parameters
    ----------
    *weights : xr.DataArray or float
        Any number of weight arrays or scalar values to combine.
        All DataArrays must be broadcastable to the same shape.
    dim : str, optional
        Dimension to normalize along. If None, will auto-detect 'time',
        'static_orbit', or 'track_pol'. If still ambiguous, normalizes over all dims.
    validate : bool
        If True, validate that final weights sum to 1.0 along dim

    Returns
    -------
    xr.DataArray
        Combined weight normalized to sum to 1.0 along the specified dimension.

    Examples
    --------
    >>> # Combine temporal and resolution weights
    >>> w_total = combine_weights(w_temporal, w_resolution, dim='time')
    >>> # w_total.sum(dim='time') == 1.0

    >>> # Combine resolution and polarization weights
    >>> w_total = combine_weights(w_resolution, w_polarization, dim='track_pol')
    >>> # w_total.sum(dim='track_pol') == 1.0

    >>> # With scalar multiplier (e.g., confidence factor)
    >>> w_total = combine_weights(w_temporal, 0.8, dim='time')
    """
    if len(weights) == 0:
        raise ValueError("Must provide at least one weight")

    # Multiply all weights together
    w_combined = reduce(operator.mul, weights)

    # Ensure result is DataArray
    if not isinstance(w_combined, xr.DataArray):
        w_combined = xr.DataArray(w_combined, name="w_total")

    # Handle NaN values (treat as zero weight)
    w_combined = w_combined.fillna(0)

    # Auto-detect dimension if not specified
    if dim is None and isinstance(w_combined, xr.DataArray):
        if 'time' in w_combined.dims:
            dim = 'time'
        elif 'static_orbit' in w_combined.dims:
            dim = 'static_orbit'
        elif 'track_pol' in w_combined.dims:
            dim = 'track_pol'
        elif len(w_combined.dims) == 1:
            dim = w_combined.dims[0]
        # If still None, will normalize over all dimensions (scalar output)

    # Normalize to sum to 1.0
    if dim is not None:
        sum_weights = w_combined.sum(dim=dim)

        # Handle case where sum is zero
        if np.any(sum_weights == 0):
            raise ValueError(
                f"Cannot normalize: sum of weights is zero along dimension '{dim}'. "
                f"Check for all-NaN or all-zero weights."
            )

        w_normalized = w_combined / sum_weights
    else:
        # Normalize over all dimensions
        sum_weights = w_combined.sum()
        if sum_weights == 0:
            raise ValueError(
                "Cannot normalize: sum of all weights is zero. "
                "Check for all-NaN or all-zero weights."
            )
        w_normalized = w_combined / sum_weights

    w_normalized = w_normalized.rename("w_total")

    return w_normalized

def weighted_mean(
    da: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    """
    Compute weighted mean over xarray dimension.
    """
    return da.weighted(weights).mean(dim)
