

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

def find_utm_crs(aoi, aoi_crs):
    """Find the UTM CRS for the center of an AOI polygon."""

    to_wgs84 = Transformer.from_crs(aoi_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_wgs84.transform(aoi.centroid.x, aoi.centroid.y)

    lon, lat = aoi.centroid.x, aoi.centroid.y
    utm_info = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(lon, lat, lon, lat),
    )[0]
    return CRS.from_authority(utm_info.auth_name, utm_info.code)


def resolution_to_meters(res, crs, lat=None):
    """
    Convert a resolution in CRS units to meters.

    Parameters
    ----------
    res : float or tuple of floats
        Resolution in CRS units (e.g., degrees if geographic, meters if projected)
        If single float, assumes same for x and y.
    crs : pyproj.CRS
        Coordinate reference system of the resolution.
    lat : float, optional
        Latitude for geographic CRS (needed to convert degrees to meters)
        Defaults to 0 (equator) if not provided.

    Returns
    -------
    (xres_m, yres_m) : tuple of floats
        Resolution in meters
    """
    # normalize
    if isinstance(res, (int, float)):
        xres, yres = float(res), float(res)
    elif isinstance(res, (tuple, list)) and len(res) == 2:
        xres, yres = float(res[0]), float(res[1])
    else:
        raise ValueError("res must be float or tuple/list of 2 floats")

    # Projected CRS (meters already)
    if crs.is_projected:
        return xres, yres

    # Geographic CRS (degrees)
    if crs.is_geographic:
        if lat is None:
            lat = 0  # default at equator
        # use pyproj transformer to go to meters
        crs_proj = CRS.from_epsg(3857)  # Web Mercator (meters)
        transformer = Transformer.from_crs(crs, crs_proj, always_xy=True)

        x0, y0 = transformer.transform(0, lat)
        x1, y1 = transformer.transform(xres, lat)
        _, y1b = transformer.transform(0, lat + yres)

        xres_m = abs(x1 - x0)
        yres_m = abs(y1b - y0)
        return xres_m, yres_m

    raise ValueError("CRS is neither projected nor geographic")

def nearest_standard_resolution(res_m, standard_res = np.array([10, 30, 90])):
    """
    Choose the closest standard SAR resolution to a given value.

    Parameters
    ----------
    res_m : float
        Resolution in meters.

    Returns
    -------
    float
        Closest resolution among 10, 30, or 90 meters.
    """
    idx = np.abs(standard_res - res_m).argmin()
    return standard_res[idx]

def resolution_to_degrees(res_m, crs, lat=None):
    """
    Convert a resolution in meters to CRS units (degrees for geographic CRS).

    Parameters
    ----------
    res_m : float or tuple of floats
        Resolution in meters (x, y)
    crs : pyproj.CRS
        Coordinate reference system to convert into.
    lat : float, optional
        Latitude at which to compute degrees if geographic CRS. Defaults to 0.

    Returns
    -------
    (xres_deg, yres_deg) : tuple of floats
        Resolution in CRS units (degrees if geographic)
    """
    # normalize input
    if isinstance(res_m, (int, float)):
        xres_m, yres_m = float(res_m), float(res_m)
    elif isinstance(res_m, (tuple, list)) and len(res_m) == 2:
        xres_m, yres_m = float(res_m[0]), float(res_m[1])
    else:
        raise ValueError("res_m must be a float or tuple/list of 2 floats")

    # projected CRS → resolution already in CRS units
    if crs.is_projected:
        return xres_m, yres_m

    # geographic CRS → convert meters to degrees
    if crs.is_geographic:
        if lat is None:
            lat = 0  # default to equator
        # Use WebMercator (EPSG:3857) as intermediate CRS
        crs_proj = CRS.from_epsg(3857)
        transformer = Transformer.from_crs(crs_proj, crs, always_xy=True)

        # Transform a point offset by xres_m, yres_m
        lon0, lat0 = 0, lat
        lon1, lat1 = transformer.transform(xres_m, 0)  # east offset
        lon2, lat2 = transformer.transform(0, yres_m)  # north offset

        xres_deg = abs(lon1 - lon0)
        yres_deg = abs(lat2 - lat0)

        return xres_deg, yres_deg

    raise ValueError("CRS is neither projected nor geographic")

def area_m2_to_pixels(da: xr.DataArray, area_m2: float) -> int:
    """
    Convert an area in square meters to number of pixels based on raster resolution.

    Parameters
    ----------
    da : xr.DataArray
        Reference raster with CRS and resolution (e.g., DEM, mask).
    area_m2 : float
        Area in square meters to convert.

    Returns
    -------
    int
        Minimum number of pixels covering the area.
    """
    # --- Get resolution ---
    res_x, res_y = np.abs(da.rio.resolution())
    # Ensure units are meters
    crs = da.rio.crs
    if crs is None:
        raise ValueError("DataArray has no CRS defined")
    if not crs.is_projected:
        raise ValueError("CRS is not projected in meters. Cannot compute pixel area.")

    pixel_area = res_x * res_y
    n_pixels = int(np.ceil(area_m2 / pixel_area))
    return n_pixels
