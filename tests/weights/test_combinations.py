import numpy as np
import pytest
import xarray as xr

from sarvalanche.weights.combinations import combine_weights, weighted_mean


# ---------------------------------------------------------------------------
# combine_weights – basic
# ---------------------------------------------------------------------------

def test_combine_two_arrays_sums_to_one():
    w1 = xr.DataArray([1.0, 2.0], dims=["time"])
    w2 = xr.DataArray([1.0, 1.0], dims=["time"])
    result = combine_weights(w1, w2, dim="time")
    assert float(result.sum()) == pytest.approx(1.0)


def test_combine_with_scalar_sums_to_one():
    w = xr.DataArray([1.0, 3.0], dims=["time"])
    result = combine_weights(w, 0.5, dim="time")
    assert float(result.sum()) == pytest.approx(1.0)


def test_combine_output_name_is_w_total():
    w = xr.DataArray([1.0, 1.0], dims=["time"])
    result = combine_weights(w, dim="time")
    assert result.name == "w_total"


def test_combine_single_weight_returns_one():
    w = xr.DataArray([1.0], dims=["time"])
    result = combine_weights(w, dim="time")
    assert float(result.values[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# combine_weights – NaN handling
# ---------------------------------------------------------------------------

def test_combine_nan_treated_as_zero():
    w = xr.DataArray([1.0, np.nan, 3.0], dims=["time"])
    result = combine_weights(w, dim="time")
    # NaN becomes 0; [1, 0, 3] normalized → [0.25, 0.0, 0.75]
    np.testing.assert_allclose(result.values, [0.25, 0.0, 0.75], rtol=1e-6)


# ---------------------------------------------------------------------------
# combine_weights – error cases
# ---------------------------------------------------------------------------

def test_combine_zero_sum_raises():
    w = xr.DataArray([0.0, 0.0], dims=["time"])
    with pytest.raises(ValueError, match="zero"):
        combine_weights(w, dim="time")


def test_combine_no_args_raises():
    with pytest.raises(ValueError, match="at least one"):
        combine_weights()


# ---------------------------------------------------------------------------
# combine_weights – auto-detect dim
# ---------------------------------------------------------------------------

def test_auto_detect_time_dim():
    w = xr.DataArray([1.0, 2.0, 3.0], dims=["time"])
    result = combine_weights(w)
    assert float(result.sum()) == pytest.approx(1.0)


def test_auto_detect_track_pol_dim():
    w = xr.DataArray([1.0, 1.0], dims=["track_pol"])
    result = combine_weights(w)
    assert float(result.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# combine_weights – value correctness
# ---------------------------------------------------------------------------

def test_combine_product_then_normalize():
    """Product of [1,2] × [3,1] = [3,2]; normalized → [0.6, 0.4]."""
    w1 = xr.DataArray([1.0, 2.0], dims=["time"])
    w2 = xr.DataArray([3.0, 1.0], dims=["time"])
    result = combine_weights(w1, w2, dim="time")
    np.testing.assert_allclose(result.values, [0.6, 0.4], rtol=1e-6)


# ---------------------------------------------------------------------------
# weighted_mean
# ---------------------------------------------------------------------------

def test_weighted_mean_matches_manual():
    da = xr.DataArray([10.0, 20.0, 30.0], dims=["time"])
    weights = xr.DataArray([0.5, 0.25, 0.25], dims=["time"])
    result = weighted_mean(da, weights, dim="time")
    expected = 10.0 * 0.5 + 20.0 * 0.25 + 30.0 * 0.25
    assert float(result) == pytest.approx(expected)


def test_weighted_mean_uniform_equals_unweighted_mean():
    da = xr.DataArray([2.0, 4.0, 6.0], dims=["time"])
    weights = xr.DataArray([1 / 3, 1 / 3, 1 / 3], dims=["time"])
    result = weighted_mean(da, weights, dim="time")
    assert float(result) == pytest.approx(float(da.mean()))
