import numpy as np
import xarray as xr
import rioxarray
from rasterio.transform import from_bounds

def make_reference_grid(
    *,
    bounds,
    crs,
    resolution,
    dtype="float32",
    fill_value=np.nan,
    name="reference",
):
    """
    Create an xarray DataArray usable as a reprojection reference grid.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in target CRS
    crs : str or CRS
        Target CRS (e.g. "EPSG:32611")
    resolution : float or (float, float)
        Pixel size in CRS units
    """

    minx, miny, maxx, maxy = bounds

    if isinstance(resolution, (int, float)):
        xres = yres = float(resolution)
    else:
        xres, yres = map(float, resolution)

    # Number of pixels
    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))

    # Affine transform (north-up)
    transform = from_bounds(
        minx, miny, minx + width * xres, miny + height * yres,
        width, height
    )

    # Pixel-centered coordinates
    x = minx + (np.arange(width) + 0.5) * xres
    y = maxy - (np.arange(height) + 0.5) * yres

    data = np.full((height, width), fill_value, dtype=dtype)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name=name,
    )

    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    return da
