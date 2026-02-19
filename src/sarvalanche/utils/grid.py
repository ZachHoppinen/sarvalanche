import numpy as np
import xarray as xr
import rioxarray

from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform
from rasterio.transform import from_bounds
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from sarvalanche.utils.constants import OPERA_RESOLUTION
from sarvalanche.utils.validation import validate_crs

def make_reference_grid(
    *,
    aoi,
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
    aoi : shapely.Polygon
        (minx, miny, maxx, maxy) in target CRS
    crs : str or CRS
        Target CRS (e.g. "EPSG:32611")
    resolution : float or (float, float)
        Pixel size in CRS units
    """

    minx, miny, maxx, maxy = aoi.bounds

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

def make_opera_reference_grid(*, aoi, aoi_crs, dtype="float32", fill_value=np.nan, name="reference"):
    """
    Create a reference grid snapped to OPERA's native 30m UTM grid.
    Automatically determines the correct UTM zone from the AOI centroid.

    Parameters
    ----------
    aoi : shapely.Polygon
        Area of interest in aoi_crs
    aoi_crs : pyproj.CRS
        CRS of the input AOI
    """

    aoi_crs= validate_crs(aoi_crs)

    # Get lon/lat of centroid
    to_wgs84 = Transformer.from_crs(aoi_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_wgs84.transform(aoi.centroid.x, aoi.centroid.y)

    # Let pyproj find the right UTM zone
    utm_info = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(lon, lat, lon, lat),
    )[0]
    utm_crs = CRS.from_authority(utm_info.auth_name, utm_info.code)

    # Reproject AOI to UTM
    transformer = Transformer.from_crs(aoi_crs, utm_crs, always_xy=True)
    aoi_utm = shapely_transform(transformer.transform, aoi)
    minx, miny, maxx, maxy = aoi_utm.bounds

    # Snap bounds to 30m OPERA grid
    minx = np.floor(minx / OPERA_RESOLUTION) * OPERA_RESOLUTION
    miny = np.floor(miny / OPERA_RESOLUTION) * OPERA_RESOLUTION
    maxx = np.ceil(maxx  / OPERA_RESOLUTION) * OPERA_RESOLUTION
    maxy = np.ceil(maxy  / OPERA_RESOLUTION) * OPERA_RESOLUTION

    width  = int(round((maxx - minx) / OPERA_RESOLUTION))
    height = int(round((maxy - miny) / OPERA_RESOLUTION))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    x = minx + (np.arange(width)  + 0.5) * OPERA_RESOLUTION
    y = maxy - (np.arange(height) + 0.5) * OPERA_RESOLUTION

    da = xr.DataArray(
        np.full((height, width), fill_value, dtype=dtype),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name=name,
    )
    da = da.rio.write_crs(utm_crs)
    da = da.rio.write_transform(transform)

    return da