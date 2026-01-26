import numpy as np
import pytest
import xarray as xr

from pathlib import Path
from datetime import datetime
from shapely.geometry import box

import rioxarray  # noqa: F401 (ensures .rio accessor is registered)


# -----------------------------------------------------------------------------
# Spatial / temporal primitives
# -----------------------------------------------------------------------------

@pytest.fixture
def aoi_projected():
    """Simple projected AOI polygon (meters)."""
    return box(500_000, 4_000_000, 501_000, 4_001_000)

@pytest.fixture
def aoi_wgs():
    # Intersects first ASF footprint (-10 to -5)
    return box(-9, -9, -4, -4)

@pytest.fixture
def start_date():
    return datetime(2024, 1, 1)


@pytest.fixture
def end_date():
    return datetime(2024, 1, 10)


@pytest.fixture
def crs():
    """Projected CRS typical for SAR processing."""
    return "EPSG:32606"


@pytest.fixture
def time_coords():
    return np.array(
        ["2024-01-01", "2024-01-07"],
        dtype="datetime64[ns]"
    )


# -----------------------------------------------------------------------------
# Canonical SAR DataArray / Dataset
# -----------------------------------------------------------------------------

@pytest.fixture
def canonical_sar_da(grid_shape, time_coords, crs):
    """
    Canonical SAR DataArray:
    dims: (time, y, x)
    """
    data = np.random.rand(len(time_coords), *grid_shape)

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": time_coords},
        name="sigma0"
    )

    da = da.rio.write_crs(crs)
    da.attrs.update(
        sensor="Sentinel-1",
        polarization="VV",
        units="linear"
    )

    return da


@pytest.fixture
def canonical_sar_ds(canonical_sar_da):
    """
    Canonical SAR Dataset wrapper.
    """
    return xr.Dataset({"sigma0": canonical_sar_da})


# -----------------------------------------------------------------------------
# DEM + terrain products
# -----------------------------------------------------------------------------

@pytest.fixture
def dem_da(grid_shape, crs):
    """DEM intentionally mismatched in resolution."""
    data = np.random.rand(grid_shape[0] // 2, grid_shape[1] // 2)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        name="elevation"
    )

    da = da.rio.write_crs(crs)
    da.attrs["units"] = "meters"

    return da


# -----------------------------------------------------------------------------
# File-system fixtures (IO)
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_dem_file(tmp_path, dem_da):
    """Write DEM to disk for IO tests."""
    path = tmp_path / "dem.tif"
    dem_da.rio.to_raster(path)
    return path


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    return out

@pytest.fixture
# Mock URLs returned by ASF
def mock_asf_urls():
    return ["https://example.com/file_VV.tif",
    "https://example.com/file_VH.tif",
    "https://example.com/file_mask.tif",
    "https://example.com/file.h5",
    "https://example.com/file_other.tif"
    ]
