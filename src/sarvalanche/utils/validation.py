# sarvalanche/utils/validation.py

import warnings
from datetime import date
from typing import Union, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from .constants import REQUIRED_ATTRS

def validate_canonical_da(
    da: xr.DataArray,
    *,
    require_time: bool | None = None,
) -> None:
    """
    Validate that a single DataArray conforms to sarvalanche's canonical model.

    Parameters
    ----------
    da : xr.DataArray
        Input array to validate.
    require_time : bool | None
        If True, require a time dimension.
        If False, forbid a time dimension.
        If None, allow either.

    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If canonical rules are violated.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray")

    # --- Dimensions ---
    if da.dims[-2:] != ("y", "x"):
        raise ValueError(f"Last two dimensions must be ('y', 'x'), got {da.dims}")

    if "time" in da.dims:
        if da.dims[0] != "time":
            raise ValueError("If present, 'time' must be the first dimension")
        time = da["time"]
        if not (np.issubdtype(time.dtype, np.datetime64) or isinstance(time.to_index(), pd.DatetimeIndex)):
            raise ValueError("'time' coordinate must be datetime64")

    if require_time is True and "time" not in da.dims:
        raise ValueError("Time dimension is required but missing")
    if require_time is False and "time" in da.dims:
        raise ValueError("Time dimension is not allowed")

    # --- Attributes ---
    missing = REQUIRED_ATTRS - set(da.attrs)
    if missing:
        raise ValueError(f"Missing required attrs: {missing}")

    # --- Mask sanity ---
    if da.dtype == bool and da.ndim not in (2, 3):
        raise ValueError("Boolean masks must be 2D or time-stacked 3D arrays")


def validate_canonical(
    data: xr.DataArray | xr.Dataset,
    *,
    require_time: bool | None = None,
) -> None:
    """
    Validate that input conforms to sarvalanche's canonical data model.
    Works for either a single DataArray or a Dataset of variables.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input to validate.
    require_time : bool | None
        Passed to individual DataArrays.
    """
    if isinstance(data, xr.DataArray):
        validate_canonical_da(data, require_time=require_time)
    elif isinstance(data, xr.Dataset):
        for name, da in data.data_vars.items():
            try:
                validate_canonical_da(da, require_time=require_time)
            except Exception as e:
                raise ValueError(f"Validation failed for variable '{name}': {e}") from e
    else:
        raise TypeError("Input must be either an xarray.DataArray or xr.Dataset")


SENTINEL_1_LAUNCH = pd.Timestamp("2014-04-03")
SENTINEL_1B_FAIL = pd.Timestamp("2021-12-23")
SENTINEL_1C_START = pd.Timestamp("2025-05-20")


def validate_dates(start_date, end_date, *, sensor: str = "Sentinel-1"):
    """
    Validate and normalize start/end dates for SAR data availability.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Normalized start and end timestamps.
    """

    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date cannot be None")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # ---- Timezone consistency ----
    if start.tz != end.tz:
        raise ValueError("start_date and end_date must share the same timezone")

    # ---- Order check ----
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    # ---- Upper bound (today) ----
    now = (
        pd.Timestamp.now(tz=start.tz)
        if start.tz is not None
        else pd.Timestamp(date.today())
    )

    if start > now or end > now:
        raise ValueError("Dates cannot be in the future")

    # ---- Sensor-specific rules ----
    if sensor.lower() in {"sentinel-1", "s1", "auto"}:
        if start < SENTINEL_1_LAUNCH:
            raise ValueError(
                "Sentinel-1 data is not available before April 2014"
            )

        # Mission health warnings (non-fatal)
        s1b_fail = SENTINEL_1B_FAIL.tz_localize(start.tz) if start.tz else SENTINEL_1B_FAIL
        s1c_start = SENTINEL_1C_START.tz_localize(start.tz) if start.tz else SENTINEL_1C_START

        if end >= s1b_fail and start < s1c_start:
            warnings.warn(
                "Date range intersects Sentinel-1B outage period "
                "(Dec 2021 → Sentinel-1C operations). "
                "Reduced revisit frequency expected.",
                UserWarning,
            )

    return start, end

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry


def validate_aoi(aoi):
    """
    Validate and normalize an AOI.

    Returns
    -------
    shapely.geometry.BaseGeometry
        Polygon or Point geometry.
    """

    geom = None

    # ---- Shapely geometry ----
    if isinstance(aoi, BaseGeometry):
        geom = aoi

    # ---- Iterable ----
    elif isinstance(aoi, (list, tuple, np.ndarray)):
        if len(aoi) == 4:
            xmin, ymin, xmax, ymax = map(float, aoi)
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            geom = box(xmin, ymin, xmax, ymax)

        elif len(aoi) == 2:
            x, y = map(float, aoi)
            geom = Point(x, y)

    # ---- Dict ----
    elif isinstance(aoi, dict):
        key_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("west", "south", "east", "north"),
            ("minx", "miny", "maxx", "maxy"),
        ]

        for keys in key_sets:
            if all(k in aoi for k in keys):
                xmin, ymin, xmax, ymax = (float(aoi[k]) for k in keys)
                xmin, xmax = sorted((xmin, xmax))
                ymin, ymax = sorted((ymin, ymax))
                geom = box(xmin, ymin, xmax, ymax)
                break

        if geom is None:
            raise ValueError(
                f"AOI dict keys not recognized: {list(aoi.keys())}. "
                f"Expected one of: {key_sets}"
            )

    else:
        raise TypeError(
            f"AOI must be geometry, iterable, or dict; got {type(aoi)}"
        )

    # ---- Geometry sanity checks ----
    if geom.is_empty:
        raise ValueError("AOI geometry is empty")

    if isinstance(geom, Polygon) and geom.area == 0:
        raise ValueError("AOI polygon has zero area")

    return geom

def within_conus(aoi):
    """
    Check whether AOI intersects approximate CONUS bounds.

    Assumes AOI is in EPSG:4326 (lon/lat).
    """
    geom = validate_aoi(aoi)
    xmin, ymin, xmax, ymax = geom.bounds

    CONUS_XMIN, CONUS_XMAX = -125.0, -66.0
    CONUS_YMIN, CONUS_YMAX = 24.0, 50.0

    return not (
        xmax < CONUS_XMIN
        or xmin > CONUS_XMAX
        or ymax < CONUS_YMIN
        or ymin > CONUS_YMAX
    )

def validate_crs(crs_input) -> CRS:
    """
    Parse a CRS input into a pyproj.CRS object with validation.

    Parameters
    ----------
    crs_input : str | int | pyproj.CRS
        CRS specification. Can be:
        - a pyproj.CRS object (returned as-is)
        - a string, e.g. "EPSG:4326" or "CRS:84"
        - an integer EPSG code, e.g. 4326

    Returns
    -------
    pyproj.CRS
        Validated CRS object.

    Raises
    ------
    ValueError
        If the input cannot be converted to a valid CRS.
    """
    # Already a pyproj CRS
    if isinstance(crs_input, CRS):
        return crs_input

    # Try integer EPSG
    if isinstance(crs_input, int):
        try:
            crs = CRS.from_epsg(crs_input)
        except Exception as e:
            raise ValueError(f"Invalid EPSG code {crs_input}: {e}")
        return crs

    # Try string
    if isinstance(crs_input, str):
        try:
            crs = CRS.from_user_input(crs_input)
        except Exception as e:
            raise ValueError(f"Invalid CRS string '{crs_input}': {e}")
        return crs

    raise TypeError(
        f"CRS must be a pyproj.CRS, string, or integer EPSG code, got {type(crs_input)}"
    )

from typing import Union, Tuple, Optional
from pyproj import CRS

def validate_resolution(
    res: Union[float, Tuple[float, float]],
    crs: Optional[CRS] = None
) -> Tuple[float, float]:
    """
    Validate and normalize resolution input, optionally checking against a CRS.

    Parameters
    ----------
    res : float or tuple of two floats
        Desired resolution. If a single float is given, it is used for both axes.
    crs : pyproj.CRS, optional
        Coordinate reference system. If provided, the resolution will be checked
        to avoid unrealistic values (e.g., a single-pixel size).

    Returns
    -------
    (xres, yres) : tuple of floats
        Resolution for x and y axes.

    Raises
    ------
    TypeError
        If input is not float or tuple of floats.
    ValueError
        If any resolution value is non-positive or obviously unreasonable for the CRS.
    """
    # Normalize input
    if isinstance(res, (int, float)):
        xres = yres = float(res)
    elif isinstance(res, (tuple, list)):
        if len(res) != 2:
            raise ValueError(f"Resolution tuple/list must have length 2, got {len(res)}")
        xres, yres = float(res[0]), float(res[1])
    else:
        raise TypeError(f"Resolution must be float or tuple of floats, got {type(res)}")

    if xres <= 0 or yres <= 0:
        raise ValueError(f"Resolution values must be positive, got {(xres, yres)}")

    # Optional CRS-based sanity check
    if crs is not None:
        if not isinstance(crs, CRS):
            raise TypeError(f"CRS must be a pyproj.CRS object, got {type(crs)}")

        # Estimate approximate "size" of one degree in meters for geographic CRS
        if crs.is_geographic:
            # Degrees; assume ~111 km per degree at equator
            deg_to_m = 111_000
            if xres < 1e-5 or yres < 1e-5:
                raise ValueError(f"Resolution {(xres, yres)} degrees is unrealistically small for CRS {crs.to_string()}")
            if xres * deg_to_m < 0.1 or yres * deg_to_m < 0.1:
                raise ValueError(f"Resolution {(xres, yres)} deg (~meters {xres*deg_to_m:.2f}, {yres*deg_to_m:.2f}) too small for CRS {crs.to_string()}")
            if xres > 10 and yres > 10:
                raise ValueError(f'Resoution  {(xres, yres)} degrees is unrealistically large for CRS {crs.to_string()}')

        # For projected CRS, we assume units are meters
        elif crs.is_projected:
            if xres < 0.01 or yres < 0.01:
                raise ValueError(f"Resolution {(xres, yres)} meters is unrealistically small for CRS {crs.to_string()}")

    return xres, yres

def validate_urls(urls: list[str], require_http: bool = True) -> list[str]:
    """
    Validate a list of URLs.

    Args:
        urls: List of strings
        require_http: Enforce that URLs start with 'http' or 'https'

    Returns:
        Cleaned list of URLs

    Raises:
        ValueError: if any URL is invalid
    """
    valid_urls = []

    for u in urls:
        if not isinstance(u, str):
            raise TypeError(f"URL must be a string, got {type(u)}: {u}")
        url = u.strip()
        if require_http and not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"Invalid URL: {url}")
        valid_urls.append(url)

    if len(valid_urls) == 0:
        raise ValueError(f'No urls found!')

    return valid_urls

def validate_path(
    filepath,
    *,
    should_exist: bool | None = None,
    make_directory: bool = False,
) -> Path:
    """
    Validate and normalize a filesystem path.

    Parameters
    ----------
    filepath : str | Path
        Input path.
    should_exist : bool | None, optional
        - True  → path must already exist
        - False → path must NOT exist
        - None  → do not enforce existence
    make_directory : bool, optional
        If True, create the directory (and parents) if it does not exist.
        Only applies when the path is intended to be a directory.

    Returns
    -------
    Path
        Normalized Path object.

    Raises
    ------
    TypeError
        If filepath cannot be interpreted as a path.
    ValueError
        If existence checks fail or directory creation is inconsistent.
    """
    try:
        path = Path(filepath).expanduser()
    except Exception as e:
        raise TypeError(f"Invalid path input: {filepath!r}") from e

    exists = path.exists()

    # --- existence checks ---
    if should_exist is True and not exists:
        raise ValueError(f"Path does not exist: {path}")

    if should_exist is False and exists:
        raise ValueError(f"Path already exists: {path}")

    # --- directory creation ---
    if make_directory:
        if exists and not path.is_dir():
            raise ValueError(
                f"Cannot create directory; path exists and is not a directory: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)

    return path
