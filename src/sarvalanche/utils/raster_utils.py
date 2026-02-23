"""
Raster processing utilities.
"""

from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

import logging
log = logging.getLogger(__name__)

def da_to01(da: xr.DataArray, old_min=0, old_max=100) -> xr.DataArray:
    """
    Normalize an xarray DataArray from [old_min, old_max] to [0, 1].
    Values outside the old range are replaced with NaN.
    """
    da = da.astype(float)  # ensure float for NaNs

    # Mask values outside the old range
    da = da.where((da >= old_min) & (da <= old_max))

    # Normalize
    if old_max == old_min:
        raise ValueError("old_max and old_min cannot be equal")

    return (da - old_min) / (old_max - old_min)

def mosaic_group(sub: xr.DataArray) -> xr.DataArray:
    """
    Combine multiple time slices into a mosaic, preserving 'time' dimension
    as the mean of the original times, and keep non-dimensional coordinates.
    """
    from functools import reduce
    import pandas as pd
    import xarray as xr
    import numpy as np

    # combine time slices
    merged = reduce(lambda a, b: a.combine_first(b), [sub.isel(time=i) for i in range(sub.sizes['time'])])

    mean_time = pd.to_datetime(sub['time']).mean()

    # Add new axis for time
    merged_data = merged.data[np.newaxis, :, :]  # shape: (1, y, x)

    # keep all coords except time
    coords = {k: v for k, v in merged.coords.items() if 'time' not in v.dims}
    coords['time'] = [mean_time]  # add the new time coordinate

    # rebuild DataArray
    merged = xr.DataArray(
        merged_data,
        dims=('time', 'y', 'x'),
        coords=coords,
        attrs=merged.attrs
    )

    # drop fully NaN rows/cols
    merged = merged.dropna('x', how='all').dropna('y', how='all')

    return merged


def combine_close_images(da, time_tol = pd.Timedelta('2min')):
    # Define tolerance
    time_tol = pd.Timedelta('2min')

    time_diff = da['time'].diff('time', label='upper')

    # Convert to NumPy, prepend zero along the 'time' axis
    data_padded = np.concatenate([[0], time_diff.values], axis=0)

    # rebuild DataArray with same 'time' coordinate
    time_diff = xr.DataArray(
        data_padded,
        dims=['time'],
        coords={'time': da['time']},
        name='time_diff'
    )

    # cumulative sum adds when over time tolerance
    groups = (time_diff >= time_tol).cumsum(dim='time')

    # group images closer than time difference
    return da.groupby(groups).map(mosaic_group)

def label_raster(da: xr.DataArray) -> xr.DataArray:
    """
    Label connected regions in a 2D DataArray using scipy.ndimage.label.
    """
    from scipy.ndimage import label as nd_label

    # Apply labeling to each time slice
    labeled = da.copy()
    if 'time' in da.dims:
        for t in range(da.sizes['time']):
            labeled.data[t] = nd_label(da.isel(time=t).data)[0]
    else:
        labeled.data = nd_label(da.data)[0]

    return labeled


def label_raster_with_aspect(da, aspect_da=None, aspect_threshold=np.pi/2):
    """
    Label connected regions in a 2D DataArray using scipy.ndimage.label.
    Optionally breaks connectivity along large aspect changes (e.g. ridgelines).

    aspect_da: DataArray of aspect in radians
    aspect_threshold: max aspect difference to allow connectivity (default 90 degrees)
    """
    from scipy.ndimage import label as nd_label

    data = da.values if hasattr(da, 'values') else da

    if aspect_da is None:
        labeled = nd_label(data)[0]
        return labeled

    aspect = aspect_da.values if hasattr(aspect_da, 'values') else aspect_da

    # compute circular difference between horizontally and vertically adjacent pixels
    diff_h = np.abs(np.arctan2(np.sin(aspect[:, :-1] - aspect[:, 1:]), 
                                np.cos(aspect[:, :-1] - aspect[:, 1:])))
    diff_v = np.abs(np.arctan2(np.sin(aspect[:-1, :] - aspect[1:, :]), 
                                np.cos(aspect[:-1, :] - aspect[1:, :])))

    # break connectivity where aspect change exceeds threshold
    connected = data.copy().astype(bool)
    connected[:, 1:]  &= diff_h < aspect_threshold
    connected[:, :-1] &= diff_h < aspect_threshold
    connected[1:, :]  &= diff_v < aspect_threshold
    connected[:-1, :] &= diff_v < aspect_threshold

    labeled = nd_label(connected)[0]
    # restore original mask — only label pixels that were active in da
    labeled[~data.astype(bool)] = 0
    return labeled