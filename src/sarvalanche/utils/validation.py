# sarvalanche/utils/validation.py

import numpy as np
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
