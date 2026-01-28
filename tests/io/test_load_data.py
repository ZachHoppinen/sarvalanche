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

from sarvalanche.io.load_datatypes import (
    # _apply_canonical_attrs,
    # _parse_rtc_timestamp,
    # load_s1_rtc,
    preallocate_output,
    read_rtc_attrs,
    load_reproject_concat_rtc,
    RTC_RAW_TO_CANONICAL,
    SENTINEL1,
    OPERA_RTC,
)

# def test_apply_canonical_attrs_maps_attributes():
#     # Dummy DataArray
#     da = xr.DataArray([1, 2, 3])
#     da.attrs = {
#         "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "dB",
#         "RADAR_BAND": "VV",
#         "PLATFORM": "SENTINEL-1",
#     }

#     da_out = _apply_canonical_attrs(da, sensor=SENTINEL1, product=OPERA_RTC, RAW_TO_CANONICAL=RTC_RAW_TO_CANONICAL)

#     # Check canonical mapping
#     assert da_out.attrs["sensor"] == SENTINEL1
#     assert da_out.attrs["product"] == OPERA_RTC
#     assert da_out.attrs["units"] == "dB"
#     assert da_out.attrs["band"] == "VV"
#     assert da_out.attrs["platform"] == "SENTINEL-1"
#     # Missing attributes should default to None
#     assert da_out.attrs["backscatter_type"] is None

# def test_parse_rtc_timestamp_standard():
#     path = Path("OPERA_L2_RTC-S1_T123-456_20200101T123456Z_S1A_30_v1.0_VV.tif")
#     ts = _parse_rtc_timestamp(path)
#     assert ts == pd.to_datetime("2020-01-01T12:34:56")

# def test_parse_rtc_timestamp_invalid_returns_none():
#     path = Path("file_without_datetime.tif")
#     ts = _parse_rtc_timestamp(path)
#     assert ts is None

# @patch("xarray.open_dataarray")
# def test_load_s1_rtc_happy_path(mock_open):
#     # Mock xarray DataArray returned by open_dataarray
#     da_mock = xr.DataArray(
#         [[[1, 2], [3, 4]]],         # shape: (band=1, y=2, x=2)
#         dims=("band", "y", "x")
#     )
#     da_mock.attrs = {"TRACK_NUMBER": "123", "ORBIT_PASS_DIRECTION": "ASCENDING", "PLATFORM": "SENTINEL-1"}
#     mock_open.return_value = da_mock

#     # Provide a valid file path
#     path = Path("OPERA_L2_RTC-S1_T123-456_20200101T123456Z_S1A_30_v1.0_VV.tif")
#     da_out = load_s1_rtc(path)

#     # Ensure xarray.open_dataarray called
#     mock_open.assert_called_once_with(path)

#     # Check canonical attributes applied
#     assert da_out.attrs["sensor"] == SENTINEL1
#     assert da_out.attrs["product"] == OPERA_RTC
#     assert da_out.attrs["track"] == "123"
#     assert da_out.attrs["direction"] == "ASCENDING"
#     assert da_out.attrs["platform"] == "SENTINEL-1"

#     # Check time dimension added
#     assert "time" in da_out.dims
#     assert da_out.time.values[0] == pd.Timestamp("2020-01-01T12:34:56")

# def test_load_s1_rtc_invalid_extension_raises():
#     path = Path("file.invalid")
#     with pytest.raises(ValueError, match="Unsupported file type"):
#         load_s1_rtc(path)


def test_preallocate_basic():
    times = ["2020-01-01", "2020-01-02"]
    y = [0, 1, 2]
    x = [10, 11]
    dtype = np.float32
    nodata = np.nan

    crs = CRS.from_epsg(4326)
    transform = Affine.translation(10, 20) * Affine.scale(1, -1)

    da = preallocate_output(times, y, x, dtype, crs, transform, nodata)

    assert isinstance(da, xr.DataArray)
    assert da.shape == (2, 3, 2)
    assert da.dtype == np.float32
    assert np.isnan(da.values).all()

def test_preallocate_spatial_metadata():
    times = [0]
    y = [0, 1]
    x = [0, 1]
    dtype = np.float32
    nodata = -9999.0

    crs = CRS.from_epsg(3857)
    transform = Affine(30, 0, -15, 0, -30, 15)

    da = preallocate_output(times, y, x, dtype, crs, transform, nodata)

    assert da.rio.crs == crs
    assert da.rio.transform() == transform
    assert da.rio.nodata == nodata

def test_preallocate_with_time_coords():
    times = ["t1", "t2", "t3"]
    y = [0]
    x = [0]
    dtype = np.float32
    nodata = np.nan

    crs = CRS.from_epsg(4326)
    transform = Affine.identity()

    time_coords = {
        "track": [12, 13, 14],
        "direction": ["ASC", "ASC", "DESC"],
        "platform": ["S1A", "S1A", "S1B"],
    }

    da = preallocate_output(
        times, y, x, dtype, crs, transform, nodata, time_coords=time_coords
    )

    for name, values in time_coords.items():
        assert name in da.coords
        assert da.coords[name].dims == ("time",)
        assert list(da.coords[name].values) == values

def test_preallocate_time_coord_length_mismatch():
    times = ["t1", "t2"]
    y = [0]
    x = [0]
    dtype = np.float32
    nodata = np.nan

    crs = CRS.from_epsg(4326)
    transform = Affine.identity()

    bad_time_coords = {
        "track": [1],  # wrong length
    }

    with pytest.raises(ValueError, match="does not match length of times"):
        preallocate_output(
            times,
            y,
            x,
            dtype,
            crs,
            transform,
            nodata,
            time_coords=bad_time_coords,
        )

def test_preallocate_integer_dtype_nodata():
    times = [0, 1]
    y = [0, 1]
    x = [0, 1]
    dtype = np.uint8
    nodata = 255

    crs = CRS.from_epsg(4326)
    transform = Affine.identity()

    da = preallocate_output(times, y, x, dtype, crs, transform, nodata)

    assert da.dtype == np.uint8
    assert np.all(da.values == nodata)
    assert da.rio.nodata == nodata

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
    out = load_reproject_concat_rtc(fps, ref_grid, pol="VV")

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
