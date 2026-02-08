# sarvalanche/utils/validation.py

import warnings
from datetime import date
from typing import Union, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from .constants import REQUIRED_ATTRS, temporal_only_vars

def check_db_linear(da: xr.DataArray):
    """
    Quick heuristic check if data is in dB or linear units.

    Returns
    -------
    str : 'dB' or 'linear'

    Notes
    -----
    - dB backscatter typically ranges from -30 to +10 dB
    - Linear backscatter typically ranges from 0 to 1 (often 0 to 0.5)
    """
    if 'units' in da.attrs:
        units = da.attrs['units'].lower()
        if 'db' in units or 'decibel' in units:
            return 'dB'
        if 'linear' in units or units == '1' or units == '':
            return 'linear'

    vmin = float(da.min())
    vmax = float(da.max())

    # If we see negative values, almost certainly dB
    if vmin < 0:
        return 'dB'

    # If max value is small (< 2), likely linear
    if vmax < 2:
        return 'linear'

    # If values span a large range (> 10), likely dB
    if vmax - vmin > 10:
        return 'dB'

    # Default guess: if all values < 5, probably linear
    return 'linear' if vmax < 5 else 'dB'


def check_rad_degrees(da: xr.DataArray):
    """
    Quick heuristic check if angles are in radians or degrees.

    Returns
    -------
    str : 'radians' or 'degrees'

    Notes
    -----
    - Radians typically range from 0 to π (3.14) or -π to π
    - Degrees typically range from 0 to 360 or -180 to 180
    - Incidence angles specifically range 0-90° or 0-1.57 rad
    """
    if 'units' in da.attrs:
        units = da.attrs['units'].lower()
        if 'rad' in units:
            return 'radians'
        if 'deg' in units:
            return 'degrees'


    vmax = float(da.max())
    vmin = float(da.min())

    # If max > π (3.15), almost certainly degrees
    if vmax > 3.15:
        return 'degrees'

    # If max < π and min >= 0, likely radians (especially if max ≈ 1.57 for incidence)
    if vmax <= 3.15 and vmin >= 0:
        return 'radians'

    # If we see values > 6.3 (2π), definitely degrees
    if vmax > 6.3:
        return 'degrees'

    # Ambiguous range (e.g., 0-3), guess based on typical incidence angle range
    # Most SAR incidence angles are 20-50°, so if max < 2, probably radians
    return 'radians' if vmax < 2 else 'degrees'

def validate_canonical_da(
    da: xr.DataArray,
    *,
    require_time: bool | None = None,
    only_time: bool = False,
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
    if da.dims[-2:] != ("y", "x") and not only_time:
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

    # --- Check for raw string coordinates ---
    for coord_name in da.coords:
        coord = da.coords[coord_name]
        if coord.dtype.kind in ['U', 'S', 'O'] and coord.size == 1:
            if isinstance(coord.data, (str, bytes, np.str_, np.bytes_)):
                raise TypeError(
                    f"Coordinate '{coord_name}' is a raw string (type: {type(coord.data).__name__}). "
                    f"String coordinates must be wrapped in numpy arrays. "
                    f"This typically happens when loading from NetCDF - use load_netcdf_to_dataset() instead of xr.open_dataset()."
                )

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
            only_time = True if name in temporal_only_vars else False
            try:
                validate_canonical_da(da, require_time=require_time, only_time=only_time)
            except Exception as e:
                raise ValueError(f"Validation failed for variable '{name}': {e}") from e
    else:
        raise TypeError("Input must be either an xarray.DataArray or xr.Dataset")


SENTINEL_1_LAUNCH = pd.Timestamp("2014-04-03")
SENTINEL_1B_FAIL = pd.Timestamp("2021-12-23")
SENTINEL_1C_START = pd.Timestamp("2025-05-20")

def validate_date(
    date: str | pd.Timestamp | np.datetime64 | datetime,
    strip_timezone: bool = True,
    timezone: str | None = None,
    allow_future: bool = False,
    param_name: str = "date",
) -> pd.Timestamp:
    """
    Validate and normalize a date input.

    Parameters
    ----------
    date : str | pd.Timestamp | np.datetime64 | datetime
        Date to validate. Can be string (ISO format), pandas Timestamp,
        numpy datetime64, or Python datetime.
    strip_timezone : bool, default=True
        If True, remove timezone information from the result.
        If False and timezone is None, preserve original timezone.
    timezone : str | None, default=None
        If provided, localize naive datetime or convert aware datetime
        to this timezone (e.g., 'UTC', 'US/Pacific').
        Ignored if strip_timezone is True.
    allow_future : bool, default=False
        If True, allow dates in the future. If False, raise error
        for dates after the current time.
    param_name : str, default='date'
        Name of parameter for error messages.

    Returns
    -------
    pd.Timestamp
        Validated and normalized timestamp.

    Raises
    ------
    ValueError
        If date cannot be parsed, is in the future (when allow_future=False),
        or is invalid.
    TypeError
        If date is of unsupported type.

    Examples
    --------
    >>> # Basic validation
    >>> validate_date('2024-03-15')
    Timestamp('2024-03-15 00:00:00')

    >>> # With timezone
    >>> validate_date('2024-03-15', strip_timezone=False, timezone='UTC')
    Timestamp('2024-03-15 00:00:00+0000', tz='UTC')

    >>> # Future date check
    >>> validate_date('2030-01-01')  # Raises ValueError

    >>> # Allow future dates
    >>> validate_date('2030-01-01', allow_future=True)
    Timestamp('2030-01-01 00:00:00')
    """

    # Try to convert to pandas Timestamp
    try:
        if isinstance(date, pd.Timestamp):
            ts = date
        elif isinstance(date, np.datetime64):
            ts = pd.Timestamp(date)
        elif isinstance(date, datetime):
            ts = pd.Timestamp(date)
        elif isinstance(date, str):
            ts = pd.to_datetime(date)
        else:
            raise TypeError(
                f"{param_name} must be string, pd.Timestamp, np.datetime64, "
                f"or datetime, got {type(date)}"
            )
    except (ValueError, pd.errors.ParserError) as e:
        raise ValueError(
            f"Could not parse {param_name}='{date}' as a valid date. "
            f"Error: {e}"
        ) from e

    # Check for NaT (Not a Time)
    if pd.isna(ts):
        raise ValueError(f"{param_name} is NaT (Not a Time)")

    # Handle timezone
    if strip_timezone:
        # Remove timezone info
        ts = ts.tz_localize(None) if ts.tz is not None else ts
    elif timezone is not None:
        # Apply requested timezone
        if ts.tz is None:
            # Naive datetime - localize
            ts = ts.tz_localize(timezone)
        else:
            # Already has timezone - convert
            ts = ts.tz_convert(timezone)

    # Check if date is in the future
    if not allow_future:
        now = pd.Timestamp.now(tz=ts.tz)
        if ts > now:
            raise ValueError(
                f"{param_name}='{ts}' is in the future. "
                f"Current time: {now}. "
                f"Set allow_future=True to allow future dates."
            )

    return ts

def validate_start_end(start_date, end_date, *, sensor: str = "Sentinel-1"):
    """
    Validate and normalize start/end dates for SAR data availability.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Normalized start and end timestamps.
    """

    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date cannot be None")

    start = validate_date(start_date)
    end = validate_date(end_date)

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