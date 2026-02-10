"""
Resolution-based weighting for SAR detection system.

Weights are normalized to sum to 1.0 along the specified dimension.
"""

import xarray as xr
import numpy as np
from typing import Optional, Union, Literal


def get_local_resolution_weights(
    local_resolution: xr.DataArray,
    dim: Optional[Union[Literal["time", "static_track"], str]] = 'static_track',
) -> xr.DataArray:
    """
    Calculate inverse resolution weights that sum to 1.0.

    Higher resolution (smaller values) receives higher weight.
    Uses standard inverse weighting: w_i = (1/r_i) / sum(1/r_j)

    Args:
        local_resolution: DataArray of resolution values (smaller = better resolution)
        dim: Dimension to normalize weights along. If None, will auto-detect
             'time' or 'static_track'
        validate: If True, validate that weights sum to 1.0

    Returns:
        DataArray of weights that sum to 1.0 along the specified dimension

    Raises:
        ValueError: If resolution contains zeros, negative values, or NaN/inf
        KeyError: If dimension not found and cannot be auto-detected

    Examples:
        >>> # Time-series with improving resolution
        >>> resolution = xr.DataArray([30, 20, 10], dims=['time'])
        >>> weights = get_local_resolution_weights(resolution)
        >>> # Results: [0.18, 0.27, 0.55] - best resolution gets highest weight

        >>> # Multiple orbits at each time
        >>> resolution = xr.DataArray(
        ...     [[30, 20], [20, 15]],
        ...     dims=['time', 'static_track']
        ... )
        >>> weights = get_local_resolution_weights(resolution, dim='time')
        >>> # Weights sum to 1.0 along time dimension for each orbit
    """

    # Validate input
    if not np.all(np.isfinite(local_resolution.values)):
        raise ValueError(
            "Resolution contains NaN or infinite values. "
            "Cannot compute weights."
        )

    if np.any(local_resolution.values <= 0):
        raise ValueError(
            "Resolution must be positive (found zero or negative values). "
            "Resolution represents physical distance - use absolute values."
        )

    # Auto-detect dimension if not specified
    if dim is None:
        if 'time' in local_resolution.dims:
            dim = 'time'
        elif 'static_track' in local_resolution.dims:
            dim = 'static_track'
        else:
            # If only one dimension, use it
            if len(local_resolution.dims) == 1:
                dim = local_resolution.dims[0]
            else:
                available_dims = list(local_resolution.dims)
                raise KeyError(
                    f"Cannot auto-detect dimension. "
                    f"Available dimensions: {available_dims}. "
                    f"Please specify dim parameter explicitly."
                )

    # Check dimension exists
    if dim not in local_resolution.dims:
        available_dims = list(local_resolution.dims)
        raise KeyError(
            f"Dimension '{dim}' not found in data. "
            f"Available dimensions: {available_dims}"
        )

    # Calculate inverse resolution (1/r)
    inverse_resolution = 1.0 / local_resolution

    # Normalize to sum to 1.0 along the specified dimension
    # w_i = (1/r_i) / sum(1/r_j)
    sum_inverse = inverse_resolution.sum(dim=dim)
    weights = inverse_resolution / sum_inverse

    # Rename for clarity
    weights = weights.rename('w_resolution')

    return weights