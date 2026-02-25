import logging
import numpy as np
import xarray as xr
from scipy.ndimage import label

log = logging.getLogger(__name__)

def filter_pixel_groups(
    da: xr.DataArray,
    min_size: int = 1,
    max_size: int = None,
    return_nlabels: bool = False,
) -> xr.DataArray:
    """
    Remove connected groups of pixels in a boolean mask that are too small or too large.

    Parameters
    ----------
    mask : xr.DataArray
        Boolean or int dataarray
    min_size : int
        Minimum number of pixels for a group to be kept.
    max_size : int or None
        Maximum number of pixels for a group to be kept. None = no upper limit.

    Returns
    -------
    xr.DataArray
        Boolean mask of same shape as input, with small/large groups removed.
    """
    arr = da.data
    # --- label connected regions ---
    labeled, n_labels = label(arr)

    # --- compute group sizes ---
    sizes = np.bincount(labeled.ravel())
    # sizes[0] is background, ignore it
    sizes[0] = 0

    nonzero_sizes = sizes[sizes > 0]
    if nonzero_sizes.size > 0:
        log.debug(
            "filter_pixel_groups: n_labels=%d, size range [%d, %d], mean=%.1f",
            n_labels, int(nonzero_sizes.min()), int(nonzero_sizes.max()), float(nonzero_sizes.mean()),
        )
    else:
        log.debug("filter_pixel_groups: n_labels=%d (no foreground regions)", n_labels)

    # --- determine which labels to remove ---
    remove_labels = np.zeros_like(sizes, dtype=bool)
    if min_size is not None:
        remove_labels |= (sizes < min_size)
    if max_size is not None:
        remove_labels |= (sizes > max_size)

    n_removed = int(remove_labels.sum())
    log.debug(
        "filter_pixel_groups: removing %d labels (min_size=%s, max_size=%s)",
        n_removed, min_size, max_size,
    )

    # --- apply mask ---
    cleaned = arr.copy()
    cleaned[np.isin(labeled, np.where(remove_labels)[0])] = False

    # --- return as xarray ---
    out = xr.DataArray(
        cleaned,
        dims=da.dims,
        coords=da.coords,
        name=f"{da.name}_filtered"
    )

    if return_nlabels:
        n_labels_post = np.sum(~remove_labels)
        return out, n_labels_post

    return out
