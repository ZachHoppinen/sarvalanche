"""Wind shelter / exposure index for avalanche release zone mapping.

Implements a vectorised version of the wind shelter algorithm from
Winstral et al. (2002) as used in AutoATES v2.0 (Veitinger et al., 2016).

For each pixel the algorithm looks upwind within a sector defined by
wind direction ± tolerance, computes terrain-angle offsets to all cells
in that sector, and takes a quantile of those angles.  Positive values
indicate sheltered (lee) terrain; negative values indicate exposed
(windward) terrain such as ridgelines.

The result is converted to a fuzzy membership via a Cauchy function
(default a=3, b=10, c=3) so that exposed ridgelines get near-zero
membership and sheltered slopes get near-one.
"""

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _pixel_size_m(da: xr.DataArray) -> float:
    """Infer pixel size in metres from coordinate spacing."""
    if da.rio.crs is not None and da.rio.crs.is_geographic:
        raise ValueError("Input must be in a projected CRS, not geographic")
    res = da.rio.resolution()
    return float(abs(res[0]))


def _sector_mask(
    radius_px: int,
    direction_deg: float,
    tolerance_deg: float,
) -> np.ndarray:
    """Build a boolean sector mask centred on (radius, radius).

    Parameters
    ----------
    radius_px : int
        Neighbourhood radius in pixels.
    direction_deg : float
        Upwind look direction in compass degrees (0=N, 90=E, …).
    tolerance_deg : float
        Half-angle of the sector.  180 = full circle.

    Returns
    -------
    np.ndarray of bool, shape (2*radius_px+1, 2*radius_px+1)
    """
    size = 2 * radius_px + 1
    y, x = np.ogrid[:size, :size]
    cy = cx = radius_px

    dy = y - cy
    dx = x - cx

    # Compass bearing from centre to each cell (0=N, clockwise)
    bearing = np.rad2deg(np.arctan2(dx, -dy)) % 360.0

    # Angular distance from look direction
    delta = (bearing - direction_deg + 180) % 360 - 180
    angle_ok = np.abs(delta) <= tolerance_deg

    # Distance mask (exclude centre, include up to radius)
    r2 = dy * dy + dx * dx
    dist_ok = (r2 > 0) & (r2 <= radius_px * radius_px)

    return (angle_ok & dist_ok).astype(bool)


def compute_wind_shelter(
    dem: xr.DataArray,
    radius_m: float = 60.0,
    direction_deg: float = 0.0,
    tolerance_deg: float = 180.0,
    quantile: float = 0.5,
) -> xr.DataArray:
    """Compute wind shelter index for every pixel.

    Vectorised implementation using ``uniform_filter`` to avoid per-pixel
    Python loops.  For each pixel, computes the terrain angle to cells
    within the upwind sector and returns the requested quantile.

    A positive value means the pixel is sheltered (surrounded by higher
    terrain upwind); negative means exposed (ridgeline / windward).

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster in a projected CRS.
    radius_m : float
        Look-distance in metres (default 60 m, as in AutoATES v2.0).
    direction_deg : float
        Prevailing wind direction in compass degrees the wind comes FROM
        (0 = north, 90 = east).  The algorithm looks upwind.
    tolerance_deg : float
        Half-angle of the upwind sector.  180 = omnidirectional (default).
    quantile : float
        Quantile of terrain angles to return (default 0.5 = median).

    Returns
    -------
    xr.DataArray
        Wind shelter index (degrees), same shape as *dem*.
    """
    pixel_m = _pixel_size_m(dem)
    radius_px = max(1, int(round(radius_m / pixel_m)))

    # Upwind direction: we look INTO the wind (opposite of wind origin)
    upwind_deg = (direction_deg + 180.0) % 360.0

    mask = _sector_mask(radius_px, upwind_deg, tolerance_deg)
    size = 2 * radius_px + 1

    # Distance from centre of kernel (metres)
    y, x = np.ogrid[:size, :size]
    cy = cx = radius_px
    dist_m = np.sqrt(((y - cy) * pixel_m) ** 2 + ((x - cx) * pixel_m) ** 2)
    dist_m[cy, cx] = np.inf  # avoid division by zero

    dem_vals = dem.values.astype(np.float64)
    nodata = ~np.isfinite(dem_vals)
    dem_filled = dem_vals.copy()
    dem_filled[nodata] = 0.0
    valid_f = (~nodata).astype(np.float64)

    n_sector = int(mask.sum())
    log.debug(
        "compute_wind_shelter: radius=%d px (%.0f m), direction=%.0f°, "
        "tolerance=%.0f°, sector pixels=%d, quantile=%.2f",
        radius_px, radius_m, direction_deg, tolerance_deg, n_sector, quantile,
    )

    if n_sector == 0:
        log.warning("compute_wind_shelter: empty sector mask, returning zeros")
        return xr.zeros_like(dem, dtype=float)

    # Strategy: for each sector pixel offset, compute the terrain angle
    # contribution across the entire raster using array shifts, then
    # aggregate with a quantile approximation.
    #
    # For exact quantile we collect all sector-pixel contributions and
    # take np.nanquantile along the stack axis.  This uses O(n_sector)
    # memory but is fully vectorised.
    rows, cols = dem_vals.shape
    sector_ys, sector_xs = np.where(mask)
    dy_offsets = sector_ys - radius_px
    dx_offsets = sector_xs - radius_px
    dists = dist_m[sector_ys, sector_xs]

    # Pre-allocate stack
    angles = np.full((n_sector, rows, cols), np.nan, dtype=np.float32)

    for k in range(n_sector):
        dy = int(dy_offsets[k])
        dx = int(dx_offsets[k])
        d = dists[k]

        # Slice source and destination regions for the shift
        src_r = slice(max(0, -dy), rows - max(0, dy))
        src_c = slice(max(0, -dx), cols - max(0, dx))
        dst_r = slice(max(0, dy), rows + min(0, dy))
        dst_c = slice(max(0, dx), cols + min(0, dx))

        elev_diff = dem_filled[src_r, src_c] - dem_filled[dst_r, dst_c]
        terrain_angle = np.rad2deg(np.arctan2(elev_diff, d))

        # Mask invalid pixels (either source or destination)
        src_valid = valid_f[src_r, src_c]
        dst_valid = valid_f[dst_r, dst_c]
        terrain_angle[~((src_valid > 0) & (dst_valid > 0))] = np.nan

        angles[k, dst_r, dst_c] = terrain_angle

    shelter = np.nanquantile(angles, quantile, axis=0)
    shelter[nodata] = np.nan

    log.debug(
        "compute_wind_shelter: result range=[%.2f, %.2f]",
        np.nanmin(shelter), np.nanmax(shelter),
    )

    return xr.DataArray(
        shelter,
        dims=dem.dims,
        coords=dem.coords,
        attrs={
            "units": "degrees",
            "source": "sarvalanche",
            "product": "wind_shelter",
            "description": (
                f"Wind shelter index (quantile={quantile}, "
                f"dir={direction_deg}°, tol={tolerance_deg}°, "
                f"radius={radius_m} m)"
            ),
        },
    )
