# sarvalanche/utils/validation.py

import warnings
from datetime import date
import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_ATTRS = {"crs", "sensor", "product", "units"}


def validate_canonical(
    da: xr.DataArray,
    *,
    require_time: bool | None = None,
) -> None:
    """
    Validate that a DataArray conforms to sarvalanche's canonical data model.

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
        raise ValueError(
            f"Last two dimensions must be ('y', 'x'), got {da.dims}"
        )

    if "time" in da.dims:
        if da.dims[0] != "time":
            raise ValueError(
                "If present, 'time' must be the first dimension"
            )
        if not np.issubdtype(da["time"].dtype, np.datetime64):
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
