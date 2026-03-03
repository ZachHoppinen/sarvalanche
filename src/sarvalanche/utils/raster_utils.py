"""
Raster processing utilities.
"""

from functools import reduce
import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
from rasterio.features import rasterize
from rasterio.transform import from_bounds

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

# ── Clip helpers ──────────────────────────────────────────────────────────────

def _y_slice(da: xr.DataArray, miny: float, maxy: float):
    """Return y slice in the correct direction for this DataArray's y coordinate."""
    y_vals = da.y.values
    if y_vals[0] > y_vals[-1]:
        return slice(maxy, miny)  # descending (UTM)
    else:
        return slice(miny, maxy)  # ascending (WGS84)

def clip_arr(da: xr.DataArray, geom) -> np.ndarray:
    minx, miny, maxx, maxy = geom.bounds
    da_local = da.sel(x=slice(minx, maxx), y=_y_slice(da, miny, maxy))
    arr = da_local.values.astype(float)
    mask = _rasterize_mask(geom, arr.shape, geom.bounds)
    arr[~mask] = np.nan
    return arr.ravel()

def clip_2d(da: xr.DataArray, geom) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = geom.bounds
    da_local = da.sel(x=slice(minx, maxx), y=_y_slice(da, miny, maxy))
    arr = da_local.values.astype(float)
    mask = _rasterize_mask(geom, arr.shape, geom.bounds)
    arr[~mask] = np.nan
    return arr, np.isfinite(arr)

def _rasterize_mask(geom, arr_shape, bounds) -> np.ndarray:
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, arr_shape[1], arr_shape[0])
    return rasterize(
        [geom], out_shape=arr_shape, transform=transform,
        dtype=np.uint8, all_touched=True,
    ).astype(bool)
# def clip_arr(da: xr.DataArray, geom) -> np.ndarray:
#     minx, miny, maxx, maxy = geom.bounds
#     da_local = da.sel(x=slice(minx, maxx), y=_y_slice(da, miny, maxy))
#     return da_local.rio.clip([geom], all_touched=True, drop=True).values.astype(float).ravel()


# def clip_2d(da: xr.DataArray, geom) -> tuple[np.ndarray, np.ndarray]:
#     minx, miny, maxx, maxy = geom.bounds
#     da_local = da.sel(x=slice(minx, maxx), y=_y_slice(da, miny, maxy))
#     arr = da_local.rio.clip([geom], all_touched=True, drop=True).values.astype(float)
#     return arr, np.isfinite(arr)

# # raster_utils.py
# def geom_bbox_clip(ds: xr.Dataset, geom) -> xr.Dataset:
#     """Coarse clip of a dataset to geometry bbox before polygon clipping.

#     Reduces data volume for subsequent rio.clip calls from full scene to
#     track extent. y slice is reversed because raster y coordinates descend.
#     """
#     minx, miny, maxx, maxy = geom.bounds
#     return ds.sel(
#         x=slice(minx, maxx),
#         y=slice(maxy, miny),
#     )


def pixel_agg_da(ds: xr.Dataset, var_list: list[str], agg: str = 'max') -> xr.DataArray | None:
    """Pixel-wise aggregation (max/mean/std) across a list of DataArrays.

    Returns None if var_list is empty.
    """
    if not var_list:
        return None
    ref = ds[var_list[0]]
    stacked = np.stack([ds[v].values.astype(float) for v in var_list], axis=0)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        agg_fns = {'max': np.nanmax, 'mean': np.nanmean, 'std': np.nanstd}
        if agg not in agg_fns:
            raise ValueError(f'Unknown agg={agg!r}')
        arr = agg_fns[agg](stacked, axis=0)
    return xr.DataArray(arr, dims=ref.dims, coords=ref.coords).rio.write_crs(ref.rio.crs)


# ── Spatial metrics ───────────────────────────────────────────────────────────

# def morans_i(arr: np.ndarray, mask: np.ndarray) -> float:
#     """Moran's I spatial autocorrelation on a 2D masked grid (rook contiguity).

#     Returns float in [-1, +1], or NaN if fewer than 3 valid pixels / zero variance.
#     """
#     valid_idx = np.argwhere(mask)
#     n = len(valid_idx)
#     if n < 3:
#         return np.nan
#     vals = arr[mask]
#     mean, var = vals.mean(), vals.var()
#     if var == 0:
#         return np.nan

#     idx_map = {(int(r), int(c)): i for i, (r, c) in enumerate(valid_idx)}
#     W_sum = cross = 0.0
#     for i, (r, c) in enumerate(valid_idx):
#         zi = vals[i] - mean
#         for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
#             j = idx_map.get((int(r) + dr, int(c) + dc))
#             if j is not None:
#                 W_sum += 1.0
#                 cross += zi * (vals[j] - mean)

#     return np.nan if W_sum == 0 else float((n / W_sum) * (cross / (n * var)))

from scipy.sparse import csr_matrix

def morans_i(arr: np.ndarray, mask: np.ndarray) -> float:
    """Moran's I spatial autocorrelation on a 2D masked grid (rook contiguity).

    Vectorised via sparse weight matrix — O(n) in valid pixels rather than
    a Python loop over neighbours.
    """
    valid_idx = np.argwhere(mask)
    n = len(valid_idx)
    if n < 3:
        return np.nan

    vals = arr[mask]
    mean = vals.mean()
    var = vals.var()
    if var == 0:
        return np.nan

    # Map (row, col) → position in vals
    idx_map = np.full(arr.shape, -1, dtype=np.intp)
    for i, (r, c) in enumerate(valid_idx):
        idx_map[r, c] = i

    # Find all rook-neighbour pairs in one vectorised pass
    rows_i, cols_j = [], []
    r, c = valid_idx[:, 0], valid_idx[:, 1]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc_ = r + dr, c + dc
        in_bounds = (nr >= 0) & (nr < arr.shape[0]) & (nc_ >= 0) & (nc_ < arr.shape[1])
        j = idx_map[nr[in_bounds], nc_[in_bounds]]
        valid = j >= 0
        rows_i.append(np.where(in_bounds)[0][valid])
        cols_j.append(j[valid])

    rows_i = np.concatenate(rows_i)
    cols_j = np.concatenate(cols_j)

    # Sparse binary weight matrix W, then compute cross term as W·z dot z
    W = csr_matrix((np.ones(len(rows_i)), (rows_i, cols_j)), shape=(n, n))
    z = vals - mean
    W_sum = W.nnz
    cross = z @ W.dot(z)

    return float((n / W_sum) * (cross / (n * var)))


def hotspot_compactness(
    arr: np.ndarray, mask: np.ndarray, threshold_pct: float = 75,
) -> dict[str, float]:
    """Connected component analysis on above-threshold pixels.

    Returns n_clusters, largest_frac, mean_cluster_dist.
    """
    from scipy.ndimage import label as nd_label
    nan_result = {'n_clusters': np.nan, 'largest_frac': np.nan, 'mean_cluster_dist': np.nan}
    vals = arr[mask]
    if vals.size < 3:
        return nan_result

    hot = (arr >= np.nanpercentile(vals, threshold_pct)) & mask
    labeled, n_clusters = nd_label(hot)
    if n_clusters == 0:
        return nan_result

    cluster_coords = [np.argwhere(labeled == lbl) for lbl in range(1, n_clusters + 1)]
    sizes = [len(c) for c in cluster_coords]
    centroids = np.array([c.mean(axis=0) for c in cluster_coords])
    mean_dist = (
        float(np.mean([
            np.linalg.norm(centroids[i] - centroids[j])
            for i, j in itertools.combinations(range(n_clusters), 2)
        ])) if n_clusters > 1 else 0.0
    )
    return {
        'n_clusters': float(n_clusters),
        'largest_frac': float(max(sizes) / sum(sizes)),
        'mean_cluster_dist': mean_dist,
    }


def effective_radius(arr: np.ndarray, mask: np.ndarray, frac: float = 0.5) -> float:
    """Radius (pixels) containing ``frac`` of total signal, weighted by value."""
    valid_idx = np.argwhere(mask)
    if len(valid_idx) < 1:
        return np.nan
    vals = arr[mask]
    w = vals - vals.min()
    total = w.sum()

    if total == 0:
        centroid = valid_idx.mean(axis=0)
        dists = np.sort(np.linalg.norm(valid_idx - centroid, axis=1))
        return float(dists[min(int(np.ceil(frac * len(dists))) - 1, len(dists) - 1)])

    centroid = (valid_idx * w[:, np.newaxis]).sum(axis=0) / total
    dists = np.linalg.norm(valid_idx - centroid, axis=1)
    order = np.argsort(dists)
    hit = min(np.searchsorted(np.cumsum(w[order]), frac * total), len(dists) - 1)
    return float(dists[order[hit]])