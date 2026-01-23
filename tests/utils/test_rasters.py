import pytest
import numpy as np
import pandas as pd
import xarray as xr
from sarvalanche.utils import raster_utils

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def sample_da():
    """Create a simple DataArray for testing."""
    data = np.array([
        [0, 50, 100],
        [25, 75, 125],  # 125 is outside old_max
    ])
    da = xr.DataArray(data, dims=["y", "x"])
    return da

@pytest.fixture
def time_da():
    """Create a 3D DataArray with 'time' dimension for mosaic/combine tests."""
    times = pd.date_range("2026-01-01", periods=3, freq="min")
    data = np.arange(3*2*2).reshape(3,2,2)
    da = xr.DataArray(data, dims=["time","y","x"], coords={"time": times})
    return da

# --------------------------
# Tests for da_to01
# --------------------------
def test_da_to01_basic(sample_da):
    out = raster_utils.da_to01(sample_da, old_min=0, old_max=100)
    expected = np.array([
        [0, 0.5, 1.0],
        [0.25, 0.75, np.nan],  # 125 > old_max → NaN
    ])
    np.testing.assert_allclose(out.values, expected, equal_nan=True)

def test_da_to01_error_on_equal_min_max(sample_da):
    with pytest.raises(ValueError):
        raster_utils.da_to01(sample_da, old_min=10, old_max=10)

# --------------------------
# Tests for mosaic_group
# --------------------------
def test_mosaic_group_basic(time_da):
    # Take the first two time slices
    sub = time_da.isel(time=[0,1])
    out = raster_utils.mosaic_group(sub)

    # Should have one time slice
    assert "time" in out.dims
    assert out.sizes["time"] == 1
    # Values from first time slice should be present where not NaN
    np.testing.assert_array_equal(out.isel(time=0).values, sub.isel(time=0).values)

# --------------------------
# Tests for combine_close_images
# --------------------------
def test_combine_close_images_groups(time_da):
    # artificially make times close together
    da = time_da.copy()
    out = raster_utils.combine_close_images(da, time_tol=pd.Timedelta("2min"))

    # Result should have fewer or same number of groups
    # Each element in the returned GroupBy map is a DataArray
    for grp in out:
        assert isinstance(grp, xr.DataArray)
        assert "time" in grp.coords
        assert grp.sizes["y"] == da.sizes["y"]
        assert grp.sizes["x"] == da.sizes["x"]

def test_combine_close_images_respects_tolerance(time_da):
    da = time_da.copy()
    # Set time difference larger than tolerance between first two
    da["time"] = pd.to_datetime(["2026-01-01T00:00:00","2026-01-01T00:05:00","2026-01-01T00:06:00"])
    out = raster_utils.combine_close_images(da, time_tol=pd.Timedelta("2min"))

    # Select the first group by the group label
    first_label = list(out['time'].values)[0]  # or groups coordinate
    first_group = out.sel(time=first_label)

    assert "time" in first_group.coords
    assert first_group["time"].ndim == 0   # scalar coordinate
