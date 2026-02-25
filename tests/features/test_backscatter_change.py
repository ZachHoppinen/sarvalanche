import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_timestep_da():
    """
    3-timestep DataArray at daily intervals.
    A timestamp between t[0] and t[1] crosses pairs: (t0→t1) and (t0→t2).
    """
    times = pd.date_range("2024-01-01", periods=3, freq="D")
    data = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times},
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_crossing_pairs_count(three_timestep_da):
    """Timestamp between t[0] and t[1] → pairs (t0→t1) and (t0→t2)."""
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert result.sizes["pair"] == 2


def test_single_crossing_pair():
    """Only two time steps; one pair crosses."""
    times = pd.date_range("2024-01-01", periods=2, freq="D")
    data = np.zeros((2, 3, 3))
    da = xr.DataArray(data, dims=["time", "y", "x"], coords={"time": times})
    result = backscatter_changes_crossing_date(da, "2024-01-01 12:00")
    assert result.sizes["pair"] == 1


def test_difference_values_correct(three_timestep_da):
    """The diff along pair=0 should equal t1 - t0 element-wise."""
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    expected_diff = three_timestep_da.isel(time=1) - three_timestep_da.isel(time=0)
    np.testing.assert_allclose(result.isel(pair=0).values, expected_diff.values)


# ---------------------------------------------------------------------------
# No crossing pairs
# ---------------------------------------------------------------------------

def test_no_crossing_pairs_raises(three_timestep_da):
    """Timestamp after all times → no crossing pairs → ValueError."""
    with pytest.raises(ValueError, match="No time pairs cross"):
        backscatter_changes_crossing_date(three_timestep_da, "2025-01-01")


def test_no_crossing_pairs_before_all_raises(three_timestep_da):
    """Timestamp before all times → no crossing pairs."""
    with pytest.raises(ValueError, match="No time pairs cross"):
        backscatter_changes_crossing_date(three_timestep_da, "2023-01-01")


# ---------------------------------------------------------------------------
# Coordinate handling
# ---------------------------------------------------------------------------

def test_t_start_coord_present(three_timestep_da):
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert "t_start" in result.coords


def test_t_end_coord_present(three_timestep_da):
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert "t_end" in result.coords


def test_t_start_before_timestamp(three_timestep_da):
    timestamp = pd.Timestamp("2024-01-01 12:00")
    result = backscatter_changes_crossing_date(three_timestep_da, timestamp)
    for ts in pd.to_datetime(result.coords["t_start"].values):
        assert ts <= timestamp


def test_t_end_after_timestamp(three_timestep_da):
    timestamp = pd.Timestamp("2024-01-01 12:00")
    result = backscatter_changes_crossing_date(three_timestep_da, timestamp)
    for te in pd.to_datetime(result.coords["t_end"].values):
        assert te > timestamp


def test_platform_coord_dropped():
    times = pd.date_range("2024-01-01", periods=3, freq="D")
    data = np.zeros((3, 2, 2))
    platforms = ["S1A", "S1B", "S1A"]
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "platform": ("time", platforms)},
    )
    result = backscatter_changes_crossing_date(da, "2024-01-01 12:00")
    assert "platform" not in result.coords


# ---------------------------------------------------------------------------
# Output name and dims
# ---------------------------------------------------------------------------

def test_output_name(three_timestep_da):
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert result.name == "delta_backscatter"


def test_output_has_pair_dim(three_timestep_da):
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert "pair" in result.dims


def test_spatial_dims_preserved(three_timestep_da):
    result = backscatter_changes_crossing_date(three_timestep_da, "2024-01-01 12:00")
    assert "y" in result.dims
    assert "x" in result.dims
