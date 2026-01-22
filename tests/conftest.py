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
def aoi():
    """Simple projected AOI polygon (meters)."""
    return box(500_000, 4_000_000, 501_000, 4_001_000)


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
def grid_shape():
    """Canonical raster shape (y, x)."""
    return (100, 120)


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


@pytest.fixture
def slope_mask(canonical_sar_da):
    """Boolean slope mask aligned to SAR grid."""
    mask = np.random.rand(
        canonical_sar_da.sizes["y"],
        canonical_sar_da.sizes["x"]
    ) > 0.2

    return xr.DataArray(
        mask,
        dims=("y", "x"),
        name="slope_mask"
    )


@pytest.fixture
def lia_mask(canonical_sar_da):
    """Local incidence angle validity mask."""
    mask = np.random.rand(
        canonical_sar_da.sizes["y"],
        canonical_sar_da.sizes["x"]
    ) > 0.1

    return xr.DataArray(
        mask,
        dims=("y", "x"),
        name="lia_mask"
    )


@pytest.fixture
def terrain_masks(slope_mask, lia_mask):
    return {
        "slope": slope_mask,
        "lia": lia_mask,
    }


# -----------------------------------------------------------------------------
# Feature outputs
# -----------------------------------------------------------------------------

@pytest.fixture
def backscatter_change(canonical_sar_da):
    """Δσ⁰ feature (time-1)."""
    diff = canonical_sar_da.diff(dim="time")

    diff.name = "delta_sigma0"
    return diff


@pytest.fixture
def coherence_change(canonical_sar_da):
    """Synthetic coherence drop feature."""
    data = np.random.rand(
        canonical_sar_da.sizes["time"] - 1,
        canonical_sar_da.sizes["y"],
        canonical_sar_da.sizes["x"]
    )

    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": canonical_sar_da.time[1:]},
        name="delta_coherence"
    )


@pytest.fixture
def feature_ds(backscatter_change, coherence_change):
    return xr.Dataset(
        {
            "delta_sigma0": backscatter_change,
            "delta_coherence": coherence_change,
        }
    )


# -----------------------------------------------------------------------------
# Detection outputs
# -----------------------------------------------------------------------------

@pytest.fixture
def detection_mask(backscatter_change):
    """Binary avalanche detection mask."""
    mask = backscatter_change > 0.8
    mask.name = "avalanche_detection"
    return mask


@pytest.fixture
def detection_ds(detection_mask):
    return xr.Dataset({"detection": detection_mask})


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
