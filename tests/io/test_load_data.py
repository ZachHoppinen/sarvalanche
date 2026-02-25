import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_origin
import rasterio

from sarvalanche.io.load_data import (
    read_rtc_attrs,
    load_reproject_concat_rtc,
    RTC_RAW_TO_CANONICAL,
    SENTINEL1,
    OPERA_RTC,
)


def test_read_rtc_attrs(tmp_path):
    # --- Create a small GeoTIFF with RTC-style tags ---
    fp = tmp_path / "test_rtc.tif"

    data = np.zeros((10, 10), dtype=np.float32)
    transform = from_origin(0, 10, 1, 1)

    tags = {
        "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "gamma0",
        "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "area",
        "RADAR_BAND": "C",
        "TRACK_NUMBER": "87",
        "ORBIT_PASS_DIRECTION": "ASCENDING",
        "PLATFORM": "SENTINEL-1A",
        "ZERO_DOPPLER_START_TIME": "2020-01-01T16:16:22.291741Z",
    }

    with rasterio.open(
        fp,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)

    # --- Run function under test ---
    attrs = read_rtc_attrs(fp)

    # --- Assertions ---
    assert attrs["units"] == "gamma0"
    assert attrs["backscatter_type"] == "area"
    assert attrs["band"] == "C"
    assert attrs["track"] == "87"
    assert attrs["direction"] == "ASCENDING"
    assert attrs["platform"] == "SENTINEL-1A"

    assert isinstance(attrs["time"], pd.Timestamp)
    assert attrs["time"] == pd.Timestamp("2020-01-01T16:16:22.291741Z")


def _write_test_rtc(fp, value, time):
    data = np.full((5, 5), value, dtype=np.float32)
    transform = from_origin(0, 5, 1, 1)

    tags = {
        "TRACK_NUMBER": "87",
        "ORBIT_PASS_DIRECTION": "ASCENDING",
        "PLATFORM": "SENTINEL-1A",
        "ZERO_DOPPLER_START_TIME": time,
    }

    with rasterio.open(
        fp,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=None,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)

def test_load_reproject_concat_rtc_simple(tmp_path):
    # --- Create two tiny input rasters ---
    fps = []
    times = [
        "2020-01-01T00:00:00Z",
        "2020-01-02T00:00:00Z",
    ]

    for i, t in enumerate(times):
        fp = tmp_path / f"rtc_{i}.tif"
        _write_test_rtc(fp, value=i + 1, time=t)
        fps.append(fp)

    # --- Reference grid (same CRS, slightly different extent) ---
    ref_data = xr.DataArray(
        np.zeros((6, 6), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.arange(6),
            "x": np.arange(6),
        },
    )
    ref_grid = ref_data.rio.write_crs("EPSG:4326")
    ref_grid = ref_grid.rio.write_transform(from_origin(0, 6, 1, 1))

    # --- Run function ---
    out = load_reproject_concat_rtc(fps, ref_grid, pol="VV", chunks="auto")

    # --- Assertions ---
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "y", "x")
    assert out.shape == (2, 6, 6)

    # time coords
    assert list(out.time.values) == [
        pd.Timestamp("2020-01-01T00:00:00Z"),
        pd.Timestamp("2020-01-02T00:00:00Z"),
    ]

    # extra coords
    assert "track" in out.coords
    assert "direction" in out.coords
    assert "platform" in out.coords

    # data was written (not all NaN)
    assert np.isfinite(out.values).any()
