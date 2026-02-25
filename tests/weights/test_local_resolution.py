import numpy as np
import pytest
import xarray as xr

from sarvalanche.weights.local_resolution import get_local_resolution_weights


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    res = xr.DataArray([30.0, 20.0, 10.0], dims=["static_track"])
    weights = get_local_resolution_weights(res, dim="static_track")
    assert float(weights.sum()) == pytest.approx(1.0)


def test_better_resolution_gets_higher_weight():
    """10 m resolution should have higher weight than 30 m."""
    res = xr.DataArray([10.0, 30.0], dims=["static_track"])
    weights = get_local_resolution_weights(res, dim="static_track")
    assert float(weights.values[0]) > float(weights.values[1])


def test_equal_resolutions_equal_weights():
    res = xr.DataArray([20.0, 20.0, 20.0], dims=["static_track"])
    weights = get_local_resolution_weights(res, dim="static_track")
    np.testing.assert_allclose(weights.values, [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)


def test_output_name_is_w_resolution():
    res = xr.DataArray([10.0, 20.0], dims=["static_track"])
    weights = get_local_resolution_weights(res, dim="static_track")
    assert weights.name == "w_resolution"


# ---------------------------------------------------------------------------
# Auto-detect dimension
# ---------------------------------------------------------------------------

def test_auto_detect_time_dim():
    res = xr.DataArray([30.0, 20.0], dims=["time"])
    weights = get_local_resolution_weights(res, dim=None)  # explicit None triggers auto-detect
    assert float(weights.sum()) == pytest.approx(1.0)


def test_auto_detect_static_track_dim():
    res = xr.DataArray([30.0, 20.0], dims=["static_track"])
    weights = get_local_resolution_weights(res)
    assert float(weights.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_zero_resolution_raises():
    res = xr.DataArray([0.0, 10.0], dims=["static_track"])
    with pytest.raises(ValueError, match="positive"):
        get_local_resolution_weights(res, dim="static_track")


def test_negative_resolution_raises():
    res = xr.DataArray([-5.0, 10.0], dims=["static_track"])
    with pytest.raises(ValueError, match="positive"):
        get_local_resolution_weights(res, dim="static_track")


def test_nan_resolution_raises():
    res = xr.DataArray([np.nan, 10.0], dims=["static_track"])
    with pytest.raises(ValueError, match="NaN"):
        get_local_resolution_weights(res, dim="static_track")


def test_inf_resolution_raises():
    res = xr.DataArray([np.inf, 10.0], dims=["static_track"])
    with pytest.raises(ValueError, match="NaN"):
        get_local_resolution_weights(res, dim="static_track")


def test_unknown_dim_raises():
    res = xr.DataArray([10.0, 20.0], dims=["static_track"])
    with pytest.raises(KeyError):
        get_local_resolution_weights(res, dim="nonexistent_dim")


# ---------------------------------------------------------------------------
# 2D input
# ---------------------------------------------------------------------------

def test_2d_weights_sum_to_one_along_dim():
    res = xr.DataArray([[10.0, 30.0], [20.0, 20.0]], dims=["time", "static_track"])
    weights = get_local_resolution_weights(res, dim="static_track")
    sums = weights.sum(dim="static_track")
    np.testing.assert_allclose(sums.values, [1.0, 1.0], rtol=1e-6)
