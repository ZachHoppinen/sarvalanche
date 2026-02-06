import xarray as xr

from functools import reduce
import operator

def combine_weights(*weights: xr.DataArray | float) -> xr.DataArray:
    """
    Combine multiple weights through multiplication.

    Parameters
    ----------
    *weights : xr.DataArray or float
        Any number of weight arrays or scalar values to combine.
        All DataArrays must be broadcastable to the same shape.

    Returns
    -------
    xr.DataArray
        Combined weight as the product of all inputs, with NaN filled as 0.

    Examples
    --------
    >>> w_total = combine_weights(w_temporal, w_stability, w_incidence)
    >>> w_total = combine_weights(w_sigma, w_lia, q_pol, names=['sigma', 'lia', 'pol'])
    """
    if len(weights) == 0:
        raise ValueError("Must provide at least one weight")

    # Multiply all weights together
    w_total = reduce(operator.mul, weights)

    # Ensure result is DataArray and handle NaN
    if not isinstance(w_total, xr.DataArray):
        w_total = xr.DataArray(w_total, name="w_total")
    else:
        w_total = w_total.fillna(0).rename("w_total")

    return w_total

def weighted_mean(
    diffs: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    """
    Compute weighted mean over xarray dimension.
    """
    return diffs.weighted(weights).mean(dim)
