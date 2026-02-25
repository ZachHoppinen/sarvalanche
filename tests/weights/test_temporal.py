import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sarvalanche.weights.temporal import get_temporal_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_times():
    """Three daily timestamps as a DataArray."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    return xr.DataArray(dates, dims=["time"])


# ---------------------------------------------------------------------------
# Single reference time (Case 2)
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_single_ref(daily_times):
    ref = pd.Timestamp("2024-01-02")
    weights = get_temporal_weights(daily_times, ref)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_closest_time_has_highest_weight(daily_times):
    # Reference is t[2] (2024-01-03); order by closeness: t2 > t1 > t0
    ref = pd.Timestamp("2024-01-03")
    weights = get_temporal_weights(daily_times, ref)
    assert weights.values[2] > weights.values[1] > weights.values[0]


def test_equidistant_times_have_equal_weights():
    times = xr.DataArray(
        pd.to_datetime(["2024-01-01", "2024-01-03"]),
        dims=["time"],
    )
    ref = pd.Timestamp("2024-01-02")  # Equidistant from both
    weights = get_temporal_weights(times, ref)
    np.testing.assert_allclose(weights.values[0], weights.values[1], rtol=1e-6)


def test_output_dim_matches_input(daily_times):
    ref = pd.Timestamp("2024-01-02")
    weights = get_temporal_weights(daily_times, ref)
    assert weights.dims == daily_times.dims


def test_output_name_is_w_temporal(daily_times):
    ref = pd.Timestamp("2024-01-02")
    weights = get_temporal_weights(daily_times, ref)
    assert weights.name == "w_temporal"


# ---------------------------------------------------------------------------
# Element-wise mode (Case 1)
# ---------------------------------------------------------------------------

def test_elementwise_equal_intervals_returns_equal_weights():
    t1 = xr.DataArray(pd.date_range("2024-01-01", periods=2, freq="D"), dims=["pair"])
    t2 = xr.DataArray(pd.date_range("2024-01-02", periods=2, freq="D"), dims=["pair"])
    weights = get_temporal_weights(t1, t2)
    np.testing.assert_allclose(weights.values, [0.5, 0.5])


def test_elementwise_weights_sum_to_one():
    t1 = xr.DataArray(pd.date_range("2024-01-01", periods=3, freq="D"), dims=["pair"])
    t2 = xr.DataArray(pd.date_range("2024-01-03", periods=3, freq="D"), dims=["pair"])
    weights = get_temporal_weights(t1, t2)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_elementwise_mismatched_length_raises():
    t1 = xr.DataArray(pd.date_range("2024-01-01", periods=2, freq="D"), dims=["t"])
    t2 = xr.DataArray(pd.date_range("2024-01-01", periods=3, freq="D"), dims=["t"])
    with pytest.raises(ValueError, match="same length"):
        get_temporal_weights(t1, t2)


# ---------------------------------------------------------------------------
# tau_days parameter
# ---------------------------------------------------------------------------

def test_larger_tau_produces_more_uniform_weights():
    times = xr.DataArray(
        pd.to_datetime(["2024-01-01", "2024-01-10"]),
        dims=["time"],
    )
    ref = pd.Timestamp("2024-01-01")

    weights_short = get_temporal_weights(times, ref, tau_days=1)
    weights_long = get_temporal_weights(times, ref, tau_days=100)

    # With short tau the closer time should dominate much more
    ratio_short = weights_short.values[0] / weights_short.values[1]
    ratio_long = weights_long.values[0] / weights_long.values[1]
    assert ratio_short > ratio_long
