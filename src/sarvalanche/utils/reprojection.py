# src/sarvalanche/utils/reprojection.py
import xarray as xr
import rioxarray
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_align(
    src: xr.DataArray,
    target: xr.DataArray,
    resampling: str = "nearest"
) -> xr.DataArray:
    """
    Reproject and align a DataArray to match the grid and CRS of a target DataArray.

    Parameters
    ----------
    src : xr.DataArray
        Source raster (e.g., DEM) to reproject.
    target : xr.DataArray
        Target raster (e.g., SAR stack) to align to.
    resampling : str
        Resampling method: 'nearest', 'bilinear', etc.

    Returns
    -------
    xr.DataArray
        Source raster reprojected and aligned to target grid.
    """
    # Convert xarray to rioxarray if needed
    if not hasattr(src, "rio"):
        src = src.rio.write_crs("EPSG:4326", inplace=True)  # default CRS if missing

    if not hasattr(target, "rio"):
        target = target.rio.write_crs("EPSG:4326", inplace=True)

    # Use rioxarray's built-in reprojection + resampling
    resampling_method = {
        "nearest": rioxarray.rioxarray.enums.Resampling.nearest,
        "bilinear": rioxarray.rioxarray.enums.Resampling.bilinear,
    }.get(resampling, rioxarray.rioxarray.enums.Resampling.nearest)

    aligned = src.rio.reproject_match(
        target,
        resampling=resampling_method
    )

    return aligned
