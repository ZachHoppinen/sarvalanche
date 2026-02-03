
import numpy as np
import xarray as xr
from scipy.ndimage import label

def filter_group_sizes(da, smallest_n_pixels, largest_n_pixels):
    labeled, n = label(da)
    # checks on n?

    sizes = np.bincount(labeled.ravel())
    remove = sizes < 8  # remove blobs smaller than 8 pixels
    remove[0] = False

    mask_clean = da.copy()
    mask_clean[remove[labeled]] = False

    out = xr.zeros_like(da)
    out.data = mask_clean

    return out