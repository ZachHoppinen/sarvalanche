import pytest

import numpy as np
import xarray as xr
from rasterio.transform import Affine

from sarvalanche.utils.grid import make_reference_grid


def test_make_reference_grid_basic():
    bounds = (500_000, 4_500_000, 500_100, 4_500_100)
    crs = "EPSG:32611"
    resolution = 10

    da = make_reference_grid(
        bounds=bounds,
        crs=crs,
        resolution=resolution,
    )

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x")

    # 100 m / 10 m = 10 pixels
    assert da.shape == (10, 10)

    assert da.rio.crs.to_string() == crs
    assert isinstance(da.rio.transform(), Affine)


def test_reference_grid_coordinates_centered_and_ordered():
    bounds = (0, 0, 100, 100)
    resolution = 10

    da = make_reference_grid(
        bounds=bounds,
        crs="EPSG:3857",
        resolution=resolution,
    )

    # x increases left → right
    assert np.all(np.diff(da.x.values) > 0)

    # y decreases top → bottom (north-up)
    assert np.all(np.diff(da.y.values) < 0)

    # Pixel centers (not edges)
    assert da.x.values[0] == 5
    assert da.y.values[0] == 95


def test_reference_grid_anisotropic_resolution():
    bounds = (0, 0, 100, 50)
    resolution = (10, 5)

    da = make_reference_grid(
        bounds=bounds,
        crs="EPSG:3857",
        resolution=resolution,
    )

    assert da.shape == (10, 10)  # 50/5, 100/10

    xres = abs(da.x.values[1] - da.x.values[0])
    yres = abs(da.y.values[1] - da.y.values[0])

    assert xres == 10
    assert yres == 5

def test_reference_grid_fill_and_dtype():
    bounds = (0, 0, 20, 20)

    da = make_reference_grid(
        bounds=bounds,
        crs="EPSG:4326",
        resolution=10,
        fill_value=-9999,
        dtype="int16",
    )

    assert da.dtype == np.int16
    assert np.all(da.values == -9999)

@pytest.fixture
def sample_da_4326():
    """Small DataArray with x/y coordinates and CRS for reproject tests"""
    data = np.ones((2, 2))
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0],
            "x": [10.0, 11.0],
        }
    )
    da = da.rio.write_crs("EPSG:4326")
    return da


def test_reference_grid_reproject_match_roundtrip(sample_da_4326):
    """
    sample_da_4326:
      - dims: y, x
      - CRS: EPSG:4326
      - has transform
    """
    ref = make_reference_grid(
        bounds=sample_da_4326.rio.bounds(),
        crs=sample_da_4326.rio.crs,
        resolution=abs(sample_da_4326.rio.resolution()[0]),
    )

    out = sample_da_4326.rio.reproject_match(ref)

    assert out.shape == ref.shape
    assert out.rio.transform() == ref.rio.transform()
    assert out.rio.crs == ref.rio.crs

def test_reference_grid_is_deterministic():
    bounds = (123, 456, 789, 1011)

    da1 = make_reference_grid(bounds=bounds, crs="EPSG:3857", resolution=30)
    da2 = make_reference_grid(bounds=bounds, crs="EPSG:3857", resolution=30)

    assert np.array_equal(da1.x, da2.x)
    assert np.array_equal(da1.y, da2.y)
    assert da1.rio.transform() == da2.rio.transform()
