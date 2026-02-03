import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, median_filter

def spatial_smooth(
    da: xr.DataArray,
    sigma: float = 1.0,
    y_dim: str = "y",
    x_dim: str = "x",
    method = 'gaussian',
    filter_size = (5, 5)
) -> xr.DataArray:
    """
    Apply Gaussian spatial smoothing to a DataArray along y and x,
    leaving other dimensions (e.g., time) unchanged.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with dims including y_dim and x_dim.
    sigma : float
        Standard deviation for Gaussian kernel (in pixels).
    y_dim : str
        Name of the y dimension.
    x_dim : str
        Name of the x dimension.

    Returns
    -------
    xr.DataArray
        Smoothed DataArray with same coords and dims as input.
    """

    # Move spatial dims to last two axes for filtering
    da_swapped = da.transpose(..., y_dim, x_dim)

    # Apply Gaussian filter slice by slice along non-spatial dims
    # We'll reshape to (..., ny, nx) to apply filter along last 2 dims
    orig_shape = da_swapped.shape
    spatial_shape = da_swapped[y_dim].size, da_swapped[x_dim].size
    other_shape = (-1,) if len(orig_shape) == 3 else orig_shape[:-2]

    data_reshaped = da_swapped.data.reshape((-1, spatial_shape[0], spatial_shape[1]))

    smoothed = np.empty_like(data_reshaped)
    for i in range(data_reshaped.shape[0]):
        if method == 'gaussian':
            smoothed[i] = gaussian_filter(data_reshaped[i], sigma=sigma, mode='nearest')
        elif method == 'median':
            smoothed[i] = median_filter(data_reshaped[i], size = filter_size)

    # reshape back to original
    smoothed = smoothed.reshape(orig_shape)

    # Create new DataArray with same coords and dims
    da_smoothed = xr.DataArray(
        smoothed,
        dims=da_swapped.dims,
        coords=da_swapped.coords,
        name=da.name
    )

    # Transpose back to original order
    da_smoothed = da_smoothed.transpose(*da.dims)

    return da_smoothed