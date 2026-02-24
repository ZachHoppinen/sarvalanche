import numpy as np
import xarray as xr
import rioxarray

from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform
from rasterio.transform import from_bounds
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import box

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

def tile_aoi(aoi, aoi_crs, max_size=0.5):
    """
    Split a shapely geometry into tiles.
    max_size is in degrees — converted to native CRS units automatically.
    """
    crs = validate_crs(aoi_crs)

    if crs.is_geographic:
        max_native = max_size  # already in degrees
    else:
        # Convert max_size degrees to metres at the centroid
        cx, cy = aoi.centroid.x, aoi.centroid.y
        to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_geo.transform(cx, cy)
        # 1 degree latitude ≈ 111320m, use that as approximation
        max_native = max_size * 111320

    minx, miny, maxx, maxy = aoi.bounds
    width  = maxx - minx
    height = maxy - miny

    if width <= max_native and height <= max_native:
        yield (0, 0), aoi
        return

    nx = int(np.ceil(width  / max_native))
    ny = int(np.ceil(height / max_native))
    tile_w = width  / nx
    tile_h = height / ny

    for ix in range(nx):
        for iy in range(ny):
            x0 = minx + ix * tile_w
            x1 = x0 + tile_w
            y0 = miny + iy * tile_h
            y1 = y0 + tile_h
            yield (ix, iy), box(x0, y0, x1, y1)

def grids_match(ds_a: xr.Dataset, ds_b: xr.Dataset) -> bool:
    """Check if two datasets share the same CRS and spatial grid."""
    crs_a = ds_a.rio.crs
    crs_b = ds_b.rio.crs
    if crs_a != crs_b:
        return False
    for coord in ("x", "y"):
        if coord not in ds_a.coords or coord not in ds_b.coords:
            return False
        if not np.allclose(ds_a[coord].values, ds_b[coord].values, rtol=1e-5):
            return False
    return True
