import numpy as np
import pytest
import xarray as xr

from sarvalanche.preprocessing.radiometric import (
    dB_to_linear,
    linear_to_dB,
    normalize_to_stable_areas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_da():
    """Simple linear-scale DataArray: [1, 10, 100] → [0, 10, 20] dB."""
    return xr.DataArray([1.0, 10.0, 100.0], dims=["x"])


@pytest.fixture
def db_da():
    """Simple dB DataArray with units attribute set."""
    return xr.DataArray([0.0, 10.0, 20.0], dims=["x"], attrs={"units": "dB"})


# ---------------------------------------------------------------------------
# linear_to_dB
# ---------------------------------------------------------------------------

def test_linear_to_dB_known_values(linear_da):
    result = linear_to_dB(linear_da)
    np.testing.assert_allclose(result.values, [0.0, 10.0, 20.0])


def test_linear_to_dB_sets_units_attr(linear_da):
    result = linear_to_dB(linear_da)
    assert result.attrs["units"] == "db"


def test_linear_to_dB_zero_becomes_nan():
    da = xr.DataArray([0.0, 1.0], dims=["x"])
    result = linear_to_dB(da)
    assert np.isnan(result.values[0])
    assert np.isfinite(result.values[1])


def test_linear_to_dB_negative_becomes_nan():
    da = xr.DataArray([-1.0, 1.0], dims=["x"])
    result = linear_to_dB(da)
    assert np.isnan(result.values[0])


def test_linear_to_dB_already_db_returns_same_object():
    da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "dB"})
    result = linear_to_dB(da)
    assert result is da


def test_linear_to_dB_nan_preserved():
    da = xr.DataArray([np.nan, 1.0], dims=["x"])
    result = linear_to_dB(da)
    assert np.isnan(result.values[0])
    assert np.isfinite(result.values[1])


def test_linear_to_dB_dims_preserved(linear_da):
    result = linear_to_dB(linear_da)
    assert result.dims == linear_da.dims


# ---------------------------------------------------------------------------
# dB_to_linear
# ---------------------------------------------------------------------------

def test_dB_to_linear_known_values(db_da):
    result = dB_to_linear(db_da)
    np.testing.assert_allclose(result.values, [1.0, 10.0, 100.0])


def test_dB_to_linear_sets_units_attr(db_da):
    result = dB_to_linear(db_da)
    assert result.attrs["units"] == "linear"


def test_dB_to_linear_already_linear_returns_same_object():
    da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "linear"})
    result = dB_to_linear(da)
    assert result is da


def test_dB_to_linear_roundtrip():
    """linear → dB → linear should recover original values."""
    original = xr.DataArray([0.1, 0.5, 1.0, 10.0], dims=["x"])
    db = linear_to_dB(original)
    db.attrs["units"] = "dB"
    restored = dB_to_linear(db)
    np.testing.assert_allclose(restored.values, original.values, rtol=1e-5)


def test_dB_to_linear_dims_preserved(db_da):
    result = dB_to_linear(db_da)
    assert result.dims == db_da.dims


# ---------------------------------------------------------------------------
# normalize_to_stable_areas
# ---------------------------------------------------------------------------

@pytest.fixture
def sar_with_offset():
    """
    Two time slices, shape (time=2, y=2, x=4).
    Stable mask covers first two x-columns.
    Time slice 1 has a +5 dB offset compared to time slice 0
    in the stable region, so normalization should remove it.
    """
    data = np.array(
        [
            # t=0: uniform values
            [[0.0, 1.0, 2.0, 3.0],
             [0.0, 1.0, 2.0, 3.0]],
            # t=1: +5 offset everywhere
            [[5.0, 6.0, 7.0, 8.0],
             [5.0, 6.0, 7.0, 8.0]],
        ]
    )
    da = xr.DataArray(data, dims=["time", "y", "x"])
    return da


@pytest.fixture
def stable_mask():
    """Boolean mask: first two x-columns are stable."""
    data = np.array(
        [[True, True, False, False],
         [True, True, False, False]]
    )
    return xr.DataArray(data, dims=["y", "x"])


def test_normalize_removes_temporal_offset(sar_with_offset, stable_mask):
    result = normalize_to_stable_areas(sar_with_offset, stable_mask)
    # After normalization the stable-area medians at each time should be equal
    stable_medians = result.where(stable_mask).median(dim=["y", "x"])
    np.testing.assert_allclose(
        stable_medians.values[0], stable_medians.values[1], atol=1e-6
    )


def test_normalize_shape_preserved(sar_with_offset, stable_mask):
    result = normalize_to_stable_areas(sar_with_offset, stable_mask)
    assert result.shape == sar_with_offset.shape
    assert result.dims == sar_with_offset.dims


def test_normalize_no_offset_unchanged(stable_mask):
    """If all time slices have the same stable-area median, output equals input."""
    data = np.zeros((3, 2, 4))
    da = xr.DataArray(data, dims=["time", "y", "x"])
    result = normalize_to_stable_areas(da, stable_mask)
    np.testing.assert_allclose(result.values, da.values)
