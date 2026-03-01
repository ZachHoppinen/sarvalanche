"""Ridgeline detection from DEM via Hessian curvature and TPI.

Derives skeletonised ridgeline masks by combining topographic position
index (TPI) high-ground filtering with principal curvature analysis.
Saddle points between ridge segments are bridged via morphological
dilation, and the result is thinned to a 1-pixel skeleton using
minimum-cost-path routing through the curvature cost surface.
"""

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    convolve,
    find_objects,
    gaussian_filter,
    label,
    uniform_filter,
)
from skimage.morphology import skeletonize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixel_size_m(dem: xr.DataArray) -> float:
    """Infer pixel size in metres from DEM coordinate spacing."""
    if dem.rio.crs is not None and dem.rio.crs.is_geographic:
        raise ValueError("DEM must be in a projected CRS, not geographic (lat/lon)")
    res = dem.rio.resolution()
    return float(abs(res[0]))


def _calculate_tpi(dem: xr.DataArray, radius_pixels: int) -> xr.DataArray:
    """Compute Topographic Position Index at a given pixel radius.

    TPI = elevation - mean(elevation in window).  Positive values indicate
    ridges/peaks, negative values indicate valleys/hollows.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster (projected CRS).
    radius_pixels : int
        Neighbourhood radius in pixels.

    Returns
    -------
    xr.DataArray
        TPI values in metres, same shape as *dem*.
    """
    window_size = 2 * radius_pixels + 1
    dem_vals = dem.values.astype(float)
    nodata_mask = ~np.isfinite(dem_vals)

    filled = dem_vals.copy()
    filled[nodata_mask] = 0.0

    window_sum = uniform_filter(filled, size=window_size, mode='nearest') * (window_size ** 2)
    window_count = uniform_filter(
        (~nodata_mask).astype(float), size=window_size, mode='nearest'
    ) * (window_size ** 2)

    mean_elev = np.where(window_count > 0, window_sum / window_count, np.nan)
    tpi_vals = dem_vals - mean_elev
    tpi_vals[nodata_mask] = np.nan

    log.debug(
        "_calculate_tpi: radius=%d px, window=%d, tpi range=[%.1f, %.1f]",
        radius_pixels, window_size, np.nanmin(tpi_vals), np.nanmax(tpi_vals),
    )

    return xr.DataArray(
        tpi_vals, dims=dem.dims, coords=dem.coords,
        attrs={'units': 'm', 'description': f'TPI radius={radius_pixels}px'},
    )


# ---------------------------------------------------------------------------
# NaN-safe gradients
# ---------------------------------------------------------------------------

def _nan_gradient_fast(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised NaN-safe gradients using np.gradient with NaN masking.

    Fills NaN pixels with zero for the gradient computation, then masks
    output pixels that have no valid neighbours in the differencing
    direction.
    """
    valid = np.isfinite(arr)

    filled = arr.copy()
    filled[~valid] = 0.0

    gy_filled, gx_filled = np.gradient(filled)

    gy = gy_filled.copy()
    gx = gx_filled.copy()

    # Mask pixels that are themselves invalid
    gy[~valid] = np.nan
    gx[~valid] = np.nan

    # Interior: mask where both neighbours in the differencing direction
    # are invalid (gradient is meaningless)
    no_y_neighbour = ~valid[2:, :] & ~valid[:-2, :]
    no_x_neighbour = ~valid[:, 2:] & ~valid[:, :-2]
    gy[1:-1, :][no_y_neighbour] = np.nan
    gx[:, 1:-1][no_x_neighbour] = np.nan

    return gy, gx


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def _gaussian_curvature(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray) -> np.ndarray:
    """Determinant of the Hessian (Gaussian curvature approximation)."""
    return gxx * gyy - gxy ** 2


def _principal_curvatures(
    gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues of the 2x2 Hessian → (lambda1, lambda2).

    lambda1 >= lambda2 by construction.
    """
    trace = gxx + gyy
    det = _gaussian_curvature(gxx, gyy, gxy)
    disc = np.sqrt(np.maximum(trace ** 2 / 4 - det, 0))
    return trace / 2 + disc, trace / 2 - disc


def compute_hessian(
    dem: xr.DataArray, smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute second-order partial derivatives of the DEM.

    Optionally applies NaN-aware Gaussian pre-smoothing to reduce noise
    in the curvature estimates.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster.
    smooth_sigma : float
        Gaussian smoothing sigma in pixels (0 to disable).

    Returns
    -------
    gxx, gyy, gxy : np.ndarray
        Second partial derivatives (d²z/dx², d²z/dy², d²z/dxdy).
    """
    vals = dem.values.astype(float)
    nodata_mask = ~np.isfinite(vals)

    if smooth_sigma > 0:
        fill = vals.copy()
        fill[nodata_mask] = 0.0
        weights = (~nodata_mask).astype(float)
        smooth_w = gaussian_filter(weights, sigma=smooth_sigma)
        vals = np.where(
            smooth_w > 0,
            gaussian_filter(fill, sigma=smooth_sigma) / smooth_w,
            np.nan,
        )

    # Only need 3 gradient calls instead of 4:
    # gx, gy from first call
    # gxx from d/dx(gx), gxy from d/dy(gx)  <- one call gets both
    # gyy from d/dy(gy)                      <- one call, discard d/dx
    gy, gx = _nan_gradient_fast(vals)
    gxy_from_gx, gxx = _nan_gradient_fast(gx)
    gyy, gyx_from_gy = _nan_gradient_fast(gy)

    # Symmetrise the cross-derivative
    gxy = np.where(
        np.isfinite(gxy_from_gx) & np.isfinite(gyx_from_gy),
        (gxy_from_gx + gyx_from_gy) / 2.0,
        np.where(np.isfinite(gxy_from_gx), gxy_from_gx, gyx_from_gy),
    )

    for arr in (gxx, gyy, gxy):
        arr[nodata_mask] = np.nan

    log.debug(
        "compute_hessian: smooth_sigma=%.1f, gxx range=[%.4f, %.4f]",
        smooth_sigma, np.nanmin(gxx), np.nanmax(gxx),
    )

    return gxx, gyy, gxy


# ---------------------------------------------------------------------------
# Morphological helpers
# ---------------------------------------------------------------------------

def _connect_within_mask(
    seeds: np.ndarray, connectable: np.ndarray, max_gap_px: int = 5,
) -> np.ndarray:
    """Dilate *seeds* up to *max_gap_px* times, constrained to *connectable*."""
    result = seeds.copy()
    for _ in range(max_gap_px):
        result = binary_dilation(result) & connectable
    log.debug(
        "_connect_within_mask: seeds=%d -> connected=%d px (max_gap=%d)",
        seeds.sum(), result.sum(), max_gap_px,
    )
    return result


def _get_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return a mask of skeleton endpoints (pixels with exactly 1 neighbour)."""
    kernel = np.ones((3, 3), dtype=int)
    nc = convolve(skel.astype(int), kernel) - skel.astype(int)
    return skel & (nc == 1)


def _mcp_skeleton_bbox(ridge_mask: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
    """Skeletonise *ridge_mask* and refine paths via minimum-cost routing.

    For each connected skeleton component, routes a minimum-cost path
    between the two most distant endpoints using *lambda2* (the smaller
    principal curvature) as the cost surface.  This removes spurious
    branches while keeping the main ridgeline.

    Parameters
    ----------
    ridge_mask : np.ndarray of bool
        Binary ridge candidate mask.
    lambda2 : np.ndarray
        Smaller principal curvature (more negative = stronger ridge).

    Returns
    -------
    np.ndarray of bool
        Refined skeleton mask.
    """
    from skimage.graph import route_through_array

    prepped = binary_closing(ridge_mask, iterations=2)
    skeleton = skeletonize(prepped)

    # Build cost surface from lambda2 (more negative = stronger ridge).
    # Use exponential scaling so strong ridges have near-zero cost and
    # weak/non-ridge pixels are exponentially more expensive.  This gives
    # the MCP much sharper contrast than linear 0-1 normalization.
    l2_masked = lambda2[prepped]
    l2_std = np.nanstd(l2_masked)
    l2_median = np.nanmedian(l2_masked)
    # Shift so that the median ridge pixel is at zero, then scale by std.
    # Negative values (strong ridge) → low cost, positive → high cost.
    normalised = (lambda2 - l2_median) / (l2_std + 1e-9)
    cost = np.exp(normalised)
    cost[~prepped] = np.inf

    result = np.zeros_like(ridge_mask, dtype=bool)
    labeled_arr, n_components = label(skeleton)
    slices = find_objects(labeled_arr)

    log.debug(
        "_mcp_skeleton_bbox: %d skeleton components, cost range=[%.3f, %.3f]",
        n_components, np.nanmin(cost[prepped]), np.nanmax(cost[prepped]),
    )

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        skel_local = (labeled_arr[sl] == i)
        ep_coords_local = np.argwhere(_get_endpoints(skel_local))

        if len(ep_coords_local) < 2:
            result[sl] |= skel_local
            continue

        # Pick the most distant endpoint pair (Euclidean) so the MCP
        # traces the full ridge rather than a short diagonal cut.
        offsets = np.array([s.start for s in sl])
        eps_global = ep_coords_local + offsets
        if len(eps_global) == 2:
            start, end = eps_global[0], eps_global[1]
        else:
            from scipy.spatial.distance import pdist, squareform
            dists = squareform(pdist(eps_global.astype(float)))
            ii, jj = np.unravel_index(dists.argmax(), dists.shape)
            start, end = eps_global[ii], eps_global[jj]

        try:
            path, _ = route_through_array(
                cost, start, end,
                fully_connected=True,
                geometric=True,
            )
            for py, px in path:
                result[py, px] = True
        except Exception:
            log.debug(
                "_mcp_skeleton_bbox: MCP failed for component %d, using raw skeleton", i,
            )
            result[sl] |= skel_local

    log.debug("_mcp_skeleton_bbox: final skeleton pixels=%d", result.sum())
    return result


# ---------------------------------------------------------------------------
# Ridge mask (internal)
# ---------------------------------------------------------------------------

def _ridge_mask(
    dem: xr.DataArray,
    tpi_radius_m: float,
    pixel_size_m: float,
    smooth_sigma: float,
    tpi_threshold: float | None,
    curv_threshold: float | None,
    max_gap_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary ridge mask from TPI + Hessian curvature.

    Returns
    -------
    connected : np.ndarray of bool
        Ridge mask with saddle-bridged connections.
    lambda2 : np.ndarray
        Smaller principal curvature (used as cost surface for skeletonisation).
    """
    tpi_radius_px = int(tpi_radius_m / pixel_size_m)
    tpi_vals = _calculate_tpi(dem, tpi_radius_px).values

    if tpi_threshold is None:
        tpi_threshold = np.nanstd(tpi_vals) * 0.3
    high_ground = tpi_vals > tpi_threshold

    log.debug(
        "_ridge_mask: tpi_radius=%d px (%.0f m), tpi_threshold=%.2f, high_ground=%d px",
        tpi_radius_px, tpi_radius_m, tpi_threshold, high_ground.sum(),
    )

    gxx, gyy, gxy = compute_hessian(dem, smooth_sigma=smooth_sigma)
    det_h = _gaussian_curvature(gxx, gyy, gxy)
    _, lambda2 = _principal_curvatures(gxx, gyy, gxy)

    if curv_threshold is None:
        curv_threshold = -np.nanstd(lambda2[high_ground]) * 0.5
    log.debug("_ridge_mask: curv_threshold=%.4f", curv_threshold)

    is_saddle = (det_h < 0) & high_ground
    is_ridge = (lambda2 <= curv_threshold) & (det_h >= 0) & high_ground

    log.debug(
        "_ridge_mask: ridge=%d px, saddle=%d px",
        is_ridge.sum(), is_saddle.sum(),
    )

    connectable = is_ridge | is_saddle
    connected = _connect_within_mask(is_ridge, connectable, max_gap_px=max_gap_px)

    return connected, lambda2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ridgelines(
    dem: xr.DataArray,
    tpi_radius_m: float = 300.0,
    pixel_size_m: float | None = None,
    smooth_sigma: float = 2.0,
    tpi_threshold: float | None = None,
    curv_threshold: float | None = None,
    max_gap_px: int = 5,
) -> xr.DataArray:
    """Derive a skeletonised ridgeline mask from a DEM.

    Parameters
    ----------
    dem : xr.DataArray
        Input elevation raster.  Must be in a projected CRS.
    tpi_radius_m : float
        Radius for coarse TPI high-ground filter in metres.
    pixel_size_m : float, optional
        DEM resolution in metres.  Inferred from *dem* if not provided.
    smooth_sigma : float
        Gaussian pre-smoothing sigma for Hessian computation (pixels).
    tpi_threshold : float, optional
        Minimum TPI value to be considered high ground.
        Defaults to ``0.3 * std(TPI)``.
    curv_threshold : float, optional
        Maximum lambda2 to be classified as a ridge.
        Defaults to ``-0.5 * std(lambda2)`` on high-ground pixels.
    max_gap_px : int
        Maximum dilation steps to bridge saddle gaps between ridge segments.

    Returns
    -------
    xr.DataArray
        Boolean ridgeline mask, same dims/coords as *dem*.
    """
    if pixel_size_m is None:
        pixel_size_m = _pixel_size_m(dem)
    log.debug(
        "generate_ridgelines: pixel_size=%.1f m, tpi_radius=%.0f m, "
        "smooth_sigma=%.1f, max_gap=%d",
        pixel_size_m, tpi_radius_m, smooth_sigma, max_gap_px,
    )

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

    log.debug(
        "generate_ridgelines: ridge_mask=%d px -> skeleton=%d px",
        ridge_mask.sum(), ridgelines.sum(),
    )

    return xr.DataArray(
        ridgelines,
        dims=dem.dims,
        coords=dem.coords,
        attrs={
            'description': 'Ridgeline skeleton',
            'units': 'bool',
            'source': 'sarvalanche',
            'product': 'ridgelines',
        },
    )
