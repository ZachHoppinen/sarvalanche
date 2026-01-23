import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from sarvalanche.io.loaders import Sentinel1RTCLoader


def test_parse_time_opera_rtc(tmp_path):
    # Example OPERA RTC file
    fname = "OPERA_L2_RTC-S1_T093-197867-IW3_20210426T013646Z_20250903T222329Z_S1A_30_v1.0_VV.tif"
    file_path = tmp_path / fname
    file_path.write_text("dummy")  # create empty file for path

    # Instantiate the loader (or DummyLoader)
    loader = Sentinel1RTCLoader()

    t = loader._parse_time(file_path)
    assert t is not None
    assert isinstance(t, pd.Timestamp)
    # Check exact timestamp from filename
    assert t == pd.Timestamp("2021-04-26T01:36:46")

@pytest.fixture
def dummy_file(tmp_path):
    """Create a minimal netCDF file mimicking a downloaded file."""
    path = tmp_path / "file.tif"
    data = np.ones((2, 2))
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        attrs={
            "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "m2",
            "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "Sigma0",
            "RADAR_BAND": "C",
            "TRACK_NUMBER": 42,
            "ORBIT_PASS_DIRECTION": "ASCENDING",
            "PLATFORM": "S1A",
        },
    )
    da.to_netcdf(path)
    return str(path)


def test_open_file_sets_canonical_attrs(dummy_file):
    loader = Sentinel1RTCLoader()
    
    with patch.object(loader, "_parse_time", return_value=pd.Timestamp("2020-01-01")):
        da = loader._open_file(dummy_file)

    # canonical attributes
    for attr in ["sensor", "product", "units", "backscatter_type", "band"]:
        assert attr in da.attrs

    # check time dimension added
    assert "time" in da.dims
    assert da.sizes["time"] == 1

    # coordinates along time
    for coord in ["track", "direction", "platform"]:
        assert coord in da.coords
        assert da.coords[coord].values[0] == da.attrs[coord]


def test_open_file_skips_time_if_parse_time_none(dummy_file):
    loader = Sentinel1RTCLoader()

    with patch.object(loader, "_parse_time", return_value=None):
        da = loader._open_file(dummy_file)

    assert "time" not in da.dims
    for coord in ["track", "direction", "platform"]:
        assert coord not in da.coords


def test_apply_canonical_attrs_fallback(dummy_file):
    loader = Sentinel1RTCLoader()
    da = xr.open_dataarray(dummy_file)

    # remove some raw attrs
    da.attrs.pop("RADAR_BAND")
    da.attrs.pop("TRACK_NUMBER")

    da2 = loader._apply_canonical_attrs(da)
    assert da2.attrs["band"] == None
    assert da2.attrs["track"] == None

@pytest.fixture
def loader():
    return Sentinel1RTCLoader()

@pytest.mark.parametrize("filename,expected", [
    ("OPERA_L2_RTC-S1_T093-197867-IW3_20210426T013646Z_20250903T222329Z_S1A_30_v1.0_VV.tif",
     pd.Timestamp("2021-04-26T01:36:46")),
    ("file_20200101_161622.tif", pd.Timestamp("2020-01-01T16:16:22")),
    ("file_20200101.tif", pd.Timestamp("2020-01-01")),
])
def test_parse_time_from_filename(loader, tmp_path, filename, expected):
    f = tmp_path / filename
    f.write_text("")  # dummy file
    assert loader._parse_time(f) == expected

def test_parse_time_raises_when_unparseable(loader, tmp_path):
    f = tmp_path / "nonsense_file.tif"
    f.write_text("")
    with pytest.raises(ValueError):
        loader._parse_time(f)

def test_open_file_creates_time_dim(loader, tmp_path):
    path = tmp_path / "file_20200101T010101Z.tif"
    da = xr.DataArray(np.ones((2,2)), dims=("y","x"), attrs={
        "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "m2",
        "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "Sigma0",
        "RADAR_BAND": "C",
        "TRACK_NUMBER": 42,
        "ORBIT_PASS_DIRECTION": "ASCENDING",
        "PLATFORM": "S1A"
    })
    da = da.rio.write_crs("EPSG:4326")
    da.rio.to_raster(path)

    # patch parse_time to ensure deterministic timestamp
    loader._parse_time = lambda p: pd.Timestamp("2020-01-01")

    out = loader._open_file(path)

    # Time dimension added
    assert "time" in out.dims
    assert out.sizes["time"] == 1

    # Coordinates along time
    for coord in ["track", "direction", "platform"]:
        assert coord in out.coords
        assert out.coords[coord].values[0] == out.attrs[coord]

def test_open_file_skips_time_if_parse_none(loader, tmp_path):
    path = tmp_path / "file.tif"
    da = xr.DataArray(np.ones((2,2)), dims=("y","x"))
    da = da.rio.write_crs("EPSG:4326")
    da.rio.to_raster(path)

    loader._parse_time = lambda p: None
    out = loader._open_file(path)
    assert "time" not in out.dims

@pytest.fixture
def tmp_tif(tmp_path):
    """Create a dummy GeoTIFF DataArray for testing."""
    path = tmp_path / "OPERA_L2_RTC-S1_T093-197867-IW3_20210426T013646Z_20250903T222329Z_S1A_30_v1.0_VV.tif"
    # Create a small 2x2 GeoTIFF using xarray + rioxarray
    da = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]}
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(path)
    return path

@pytest.fixture
def tmp_nc(tmp_path):
    """Create a dummy NetCDF file to test extension rejection."""
    path = tmp_path / "test.nc"
    da = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]}
    )
    da.to_netcdf(path)
    return path

def test_open_file_tif(tmp_tif):
    loader = Sentinel1RTCLoader()
    da = loader._open_file(tmp_tif)
    assert isinstance(da, xr.DataArray)
    assert "time" in da.dims

def test_open_file_invalid_extension(tmp_nc):
    loader = Sentinel1RTCLoader()
    with pytest.raises(ValueError, match="Sentinel1RTCLoader only supports GeoTIFF files"):
        loader._open_file(tmp_nc)
