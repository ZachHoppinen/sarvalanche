
from functools import reduce
import operator
import logging

import xarray as xr

# probability functions
from sarvalanche.weights.local_resolution import get_local_resolution_weights
from sarvalanche.weights.temporal import get_temporal_weights

log = logging.getLogger(__name__)

def get_static_weights(ds, avalanche_date):

    ds['w_resolution'] = get_local_resolution_weights(ds['anf'])
    ds['w_temporal'] = get_temporal_weights(ds['time'], avalanche_date)

    for d in ['w_resolution', 'w_temporal']:
        ds[d].attrs = {'source': 'sarvalance', 'units': '1', 'product': 'weight'}

    return ds


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
    da: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    """
    Compute weighted mean over xarray dimension.
    """
    return da.weighted(weights).mean(dim)
