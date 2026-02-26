import logging
import numpy as np
import xarray as xr
from affine import Affine
from scipy.ndimage import uniform_filter

log = logging.getLogger(__name__)


def _assert_projected(dem, fn_name):
    """Raise ValueError if dem has no CRS or a geographic (non-projected) CRS."""
    crs = dem.rio.crs
    if crs is None:
        raise ValueError(f"{fn_name}: DataArray has no CRS defined")
    if not crs.is_projected:
        raise ValueError(
            f"{fn_name}: CRS must be projected (got {crs}). "
            "Reproject the DEM to a metric CRS before calling this function."
        )


def _tpi_px(dem, radius_px):
    """
    Compute TPI at a given pixel radius (internal helper, no CRS check).

    NaN nodata cells are excluded from the neighbourhood mean by zeroing them
    before filtering and restoring NaN afterwards.
    """
    size = 2 * radius_px + 1
    arr = dem.values.copy().astype(float)
    nodata_mask = ~np.isfinite(arr)
    arr[nodata_mask] = 0.0
    mean_arr = uniform_filter(arr, size=size, mode="nearest")
    tpi_arr = dem.values.astype(float) - mean_arr
    tpi_arr[nodata_mask] = np.nan
    return tpi_arr


def compute_tpi(dem, radius_m=300.0):
    """
    Compute the Topographic Position Index (TPI).

    TPI = elevation − mean elevation within a circular neighbourhood of
    radius ``radius_m`` metres.  Positive values indicate ridges/peaks;
    negative values indicate valleys/hollows; near-zero indicates
    mid-slope or flat terrain.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster.  Must have a projected CRS.
    radius_m : float
        Neighbourhood radius in metres (default 300 m).

    Returns
    -------
    xr.DataArray
        TPI values with the same shape/coords as *dem*, named ``"tpi"``.
    """
    _assert_projected(dem, "compute_tpi")
    res_x = abs(dem.rio.resolution()[0])
    radius_px = max(1, round(radius_m / res_x))
    return xr.DataArray(_tpi_px(dem, radius_px), dims=dem.dims, coords=dem.coords, name="tpi")


def multiscale_ridgeline_tpi(
    dem: xr.DataArray,
    fine_radius_m: float = 200.0,
    coarse_radius_m: float = 1000.0,
    ridge_threshold: float = 2.0,
    coarse_min: float = -5.0,
) -> dict:
    """
    Identify ridgeline pixels using two-scale TPI.

    A pixel is classified as a ridgeline when it is:

    * Locally elevated at the fine scale  (``tpi_fine > ridge_threshold``) —
      picks up ridge crests and spurs.
    * Not deeply negative at the coarse scale (``tpi_coarse > coarse_min``) —
      rejects valley-side bumps that look locally high but sit inside a
      large valley.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster.  Must have a projected CRS.
    fine_radius_m : float
        Neighbourhood radius for the fine-scale TPI (default 200 m).
    coarse_radius_m : float
        Neighbourhood radius for the coarse-scale TPI (default 1000 m).
    ridge_threshold : float
        Minimum fine-scale TPI (metres) to qualify as a ridgeline pixel
        (default 2.0 m).
    coarse_min : float
        Minimum coarse-scale TPI (metres) allowed — pixels below this are
        considered to lie inside a large valley (default −5.0 m).

    Returns
    -------
    dict with keys:
        ``tpi_fine``       — xr.DataArray of fine-scale TPI
        ``tpi_coarse``     — xr.DataArray of coarse-scale TPI
        ``ridgeline_mask`` — boolean xr.DataArray (True = ridgeline pixel)
    """
    _assert_projected(dem, "multiscale_ridgeline_tpi")

    res_x = abs(dem.rio.resolution()[0])
    fine_px   = max(1, round(fine_radius_m   / res_x))
    coarse_px = max(1, round(coarse_radius_m / res_x))

    tpi_fine_arr   = _tpi_px(dem, fine_px)
    tpi_coarse_arr = _tpi_px(dem, coarse_px)

    tpi_fine   = xr.DataArray(tpi_fine_arr,   dims=dem.dims, coords=dem.coords, name="tpi_fine")
    tpi_coarse = xr.DataArray(tpi_coarse_arr, dims=dem.dims, coords=dem.coords, name="tpi_coarse")

    ridgeline_mask = ((tpi_fine > ridge_threshold) & (tpi_coarse > coarse_min)).astype(float)

    n_ridge  = int(ridgeline_mask.values.sum())
    n_valid  = int(np.isfinite(tpi_fine_arr).sum())
    pct      = 100.0 * n_ridge / n_valid if n_valid > 0 else 0.0

    log.debug(
        "multiscale_ridgeline_tpi: fine TPI (%gm / %dpx) range [%.1f, %.1f] m",
        fine_radius_m, fine_px,
        float(np.nanmin(tpi_fine_arr)), float(np.nanmax(tpi_fine_arr)),
    )
    log.debug(
        "multiscale_ridgeline_tpi: coarse TPI (%gm / %dpx) range [%.1f, %.1f] m",
        coarse_radius_m, coarse_px,
        float(np.nanmin(tpi_coarse_arr)), float(np.nanmax(tpi_coarse_arr)),
    )
    log.debug(
        "multiscale_ridgeline_tpi: %d ridgeline pixels (%.1f%% of valid)",
        n_ridge, pct,
    )

    return {
        "tpi_fine":       tpi_fine,
        "tpi_coarse":     tpi_coarse,
        "ridgeline_mask": ridgeline_mask,
    }


def compute_curvature(dem):
    """
    Compute total curvature of a DEM using xrspatial.

    Units: 1/100 z-unit (as returned by xrspatial.curvature).  Positive
    values indicate convex (hill) surfaces; negative values indicate
    concave (valley) surfaces.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster.  Must have a projected CRS.

    Returns
    -------
    xr.DataArray
        Curvature values.  Outer border pixels will be NaN (xrspatial
        behaviour).
    """
    _assert_projected(dem, "compute_curvature")

    import xrspatial

    dem_copy = dem.copy()
    res_x, res_y = dem.rio.resolution()
    dem_copy.attrs["res"] = (abs(res_x), abs(res_y))

    return xrspatial.curvature(dem_copy)


def compute_flow_accumulation(dem):
    """
    Compute D8 flow accumulation using pysheds.

    Each cell value is the number of upstream cells (including itself)
    that drain into it.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster.  Must have a projected CRS.

    Returns
    -------
    xr.DataArray
        Flow accumulation values with the same shape/coords as *dem*,
        named ``"flow_accumulation"``.  NaN is set at original nodata
        positions.
    """
    _assert_projected(dem, "compute_flow_accumulation")

    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    NODATA = -9999.0

    arr = dem.values.copy().astype(float)
    nodata_mask = ~np.isfinite(arr)
    arr[nodata_mask] = NODATA

    t = dem.rio.transform()
    affine = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
    vf = ViewFinder(affine=affine, shape=arr.shape, nodata=NODATA)
    raster = Raster(arr, viewfinder=vf)

    grid = Grid.from_raster(raster)
    pit_filled = grid.fill_pits(raster)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)

    acc_arr = np.array(acc, dtype=float)
    acc_arr[nodata_mask] = np.nan

    return xr.DataArray(
        acc_arr, dims=dem.dims, coords=dem.coords, name="flow_accumulation"
    )
