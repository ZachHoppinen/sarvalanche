
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter, binary_closing, binary_dilation, label, convolve
from skimage.morphology import skeletonize
# from skimage.graph import route_through_array

def calculate_tpi(dem: xr.DataArray, radius_pixels: int) -> xr.DataArray:
    window_size = 2 * radius_pixels + 1
    dem_vals = dem.values.astype(float)
    nodata_mask = ~np.isfinite(dem_vals)

    filled = dem_vals.copy()
    filled[nodata_mask] = 0.0

    window_sum   = uniform_filter(filled, size=window_size, mode='nearest') * (window_size ** 2)
    window_count = uniform_filter((~nodata_mask).astype(float), size=window_size, mode='nearest') * (window_size ** 2)

    mean_elev = np.where(window_count > 0, window_sum / window_count, np.nan)
    tpi_vals  = dem_vals - mean_elev
    tpi_vals[nodata_mask] = np.nan

    return xr.DataArray(tpi_vals, dims=dem.dims, coords=dem.coords,
                        attrs={'units': 'm', 'description': f'TPI radius={radius_pixels}px'})



def _mcp_skeleton_bbox(ridge_mask: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
    from skimage.graph import route_through_array
    from scipy.ndimage import binary_closing, label, find_objects
    from skimage.morphology import skeletonize

    prepped  = binary_closing(ridge_mask, iterations=2)
    skeleton = skeletonize(prepped)

    l2_min, l2_max = np.nanmin(lambda2), np.nanmax(lambda2)
    cost = (lambda2 - l2_min) / (l2_max - l2_min + 1e-9)
    cost[~prepped] = np.inf

    def get_endpoints(skel: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=int)
        nc = convolve(skel.astype(int), kernel) - skel.astype(int)
        return skel & (nc == 1)

    result  = np.zeros_like(ridge_mask, dtype=bool)
    labeled, n = label(skeleton)
    slices = find_objects(labeled)

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        # Get endpoints from the local skeleton crop
        skel_local = (labeled[sl] == i)
        ep_coords_local = np.argwhere(get_endpoints(skel_local))

        if len(ep_coords_local) < 2:
            result[sl] |= skel_local
            continue

        # Translate local coords back to full array coords
        offsets = np.array([s.start for s in sl])
        start = ep_coords_local[0]  + offsets
        end   = ep_coords_local[-1] + offsets

        # Route on the FULL cost array — no crop, no padding needed
        try:
            path, _ = route_through_array(
                cost, start, end,
                fully_connected=True,
                geometric=True,
            )
            for py, px in path:
                result[py, px] = True
        except Exception:
            result[sl] |= skel_local

    return result



def gaussian_curvature(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray) -> np.ndarray:
    return gxx * gyy - gxy ** 2


def principal_curvatures(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trace = gxx + gyy
    det   = gaussian_curvature(gxx, gyy, gxy)
    disc  = np.sqrt(np.maximum(trace ** 2 / 4 - det, 0))
    return trace / 2 + disc, trace / 2 - disc   # lambda1, lambda2

def _connect_within_mask(seeds: np.ndarray, connectable: np.ndarray, max_gap_px: int = 5) -> np.ndarray:
    result = seeds.copy()
    for _ in range(max_gap_px):
        result = binary_dilation(result) & connectable
    return result


def _nan_gradient_fast(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised NaN-safe gradients using np.gradient with NaN masking.
    Much faster than the boundary-loop version for large arrays.
    """
    valid = np.isfinite(arr)
    
    # Fill NaNs with local mean for gradient computation
    # This avoids NaN propagation without per-pixel logic
    filled = arr.copy()
    filled[~valid] = 0.0
    
    # Compute gradients on filled array
    gy_filled, gx_filled = np.gradient(filled)
    
    # Compute gradients on validity mask to detect boundary influence
    gy_valid, gx_valid = np.gradient(valid.astype(float))
    
    # Where validity changes (boundary), gradient is unreliable — mask it
    # Use central diff on valid mask: if neighbours differ in validity, 
    # fall back to NaN
    valid_y = valid[2:, :] | valid[:-2, :]   # at least one valid neighbour in y
    valid_x = valid[:, 2:] | valid[:, :-2]   # at least one valid neighbour in x
    
    gy = gy_filled.copy()
    gx = gx_filled.copy()
    
    # Mask where no valid neighbours exist
    gy[~valid] = np.nan
    gx[~valid] = np.nan
    
    # Interior: mask where both neighbours are invalid
    no_y_neighbour = ~valid[2:, :] & ~valid[:-2, :]
    no_x_neighbour = ~valid[:, 2:] & ~valid[:, :-2]
    gy[1:-1, :][no_y_neighbour] = np.nan
    gx[:, 1:-1][no_x_neighbour] = np.nan

    return gy, gx

def compute_hessian_fast(dem: xr.DataArray, smooth_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = dem.values.astype(float)
    nodata_mask = ~np.isfinite(vals)

    if smooth_sigma > 0:
        fill    = vals.copy()
        fill[nodata_mask] = 0.0
        weights = (~nodata_mask).astype(float)
        smooth_w = gaussian_filter(weights, sigma=smooth_sigma)
        vals = np.where(smooth_w > 0,
                        gaussian_filter(fill, sigma=smooth_sigma) / smooth_w,
                        np.nan)

    # Only need 3 gradient calls instead of 4:
    # gx, gy from first call
    # gxx from d/dx(gx), gxy from d/dy(gx)  <- one call gets both
    # gyy from d/dy(gy)                      <- one call, discard d/dx
    gy, gx = _nan_gradient_fast(vals)
    gxy_from_gx, gxx = _nan_gradient_fast(gx)   # d/dy(gx), d/dx(gx)
    gyy, gyx_from_gy = _nan_gradient_fast(gy)   # d/dy(gy), d/dx(gy)

    # Symmetrise cross term
    gxy = np.where(
        np.isfinite(gxy_from_gx) & np.isfinite(gyx_from_gy),
        (gxy_from_gx + gyx_from_gy) / 2.0,
        np.where(np.isfinite(gxy_from_gx), gxy_from_gx, gyx_from_gy)
    )

    for arr in (gxx, gyy, gxy):
        arr[nodata_mask] = np.nan

    return gxx, gyy, gxy

def _ridge_mask(
    dem: xr.DataArray,
    tpi_radius_m: float,
    pixel_size_m: float,
    smooth_sigma: float,
    tpi_threshold: float | None,
    curv_threshold: float | None,
    max_gap_px: int,
    _hessian_fn=compute_hessian_fast
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ridge_mask, lambda2)."""
    tpi_radius_px = int(tpi_radius_m / pixel_size_m)
    tpi_vals      = calculate_tpi(dem, tpi_radius_px).values

    if tpi_threshold is None:
        tpi_threshold = np.nanstd(tpi_vals) * 0.3
    high_ground = tpi_vals > tpi_threshold

    gxx, gyy, gxy = _hessian_fn(dem, smooth_sigma=smooth_sigma)
    det_h         = gaussian_curvature(gxx, gyy, gxy)
    _, lambda2    = principal_curvatures(gxx, gyy, gxy)

    if curv_threshold is None:
        curv_threshold = -np.nanstd(lambda2[high_ground]) * 0.5

    is_saddle = (det_h < 0)                                              & high_ground
    is_ridge  = (lambda2 <= curv_threshold) & (det_h >= 0)              & high_ground

    connectable = is_ridge | is_saddle
    connected   = _connect_within_mask(is_ridge, connectable, max_gap_px=max_gap_px)

    return connected, lambda2

def generate_ridgelines(
    dem: xr.DataArray,
    tpi_radius_m: float = 300,
    pixel_size_m: float = cell_size,
    smooth_sigma: float = 2.0,
    tpi_threshold: float | None = None,
    curv_threshold: float | None = None,
    max_gap_px: int = 5,
) -> xr.DataArray:
    """
    Derive a skeletonised ridgeline mask from a DEM.

    Parameters
    ----------
    dem             Input elevation DataArray.
    tpi_radius_m    Radius for coarse TPI zone mask in metres.
    pixel_size_m    DEM resolution in metres.
    smooth_sigma    Gaussian pre-smoothing sigma for Hessian (pixels).
    tpi_threshold   Minimum TPI value to be considered high ground.
                    Defaults to 0.3 * std(TPI).
    curv_threshold  Maximum lambda2 to be classified as a ridge.
                    Defaults to -0.5 * std(lambda2 on high ground).
    max_gap_px      How many dilation steps to bridge saddle gaps.

    Returns
    -------
    xr.DataArray    Boolean array, same dims/coords as dem.
                    True = ridgeline pixel.
    """
    ridge_mask, lambda2 = _ridge_mask(
        dem,
        tpi_radius_m=tpi_radius_m,
        pixel_size_m=pixel_size_m,
        smooth_sigma=smooth_sigma,
        tpi_threshold=tpi_threshold,
        curv_threshold=curv_threshold,
        max_gap_px=max_gap_px,
    )

    ridgelines = _mcp_skeleton_bbox(ridge_mask, lambda2)
    # ridgelines = _mcp_skeleton(ridge_mask, lambda2)

    return xr.DataArray(
        ridgelines,
        dims=dem.dims,
        coords=dem.coords,
        attrs={'description': 'Ridgeline skeleton', 'units': 'bool'},
    )