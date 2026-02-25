import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sarvalanche.probabilities.features import (
    probability_backscatter_change,
    probability_cell_counts,
    probability_forest_cover,
    probability_slope_angle,
    probability_swe_accumulation,
)


# ---------------------------------------------------------------------------
# probability_backscatter_change
# ---------------------------------------------------------------------------

def test_backscatter_at_threshold_is_half():
    """At the default threshold (1.5 dB), probability should be ~0.5."""
    diff = xr.DataArray([1.5], dims=["x"])
    result = probability_backscatter_change(diff)
    assert float(result.values[0]) == pytest.approx(0.5, abs=1e-4)


def test_backscatter_above_threshold_gt_half():
    diff = xr.DataArray([5.0], dims=["x"])
    result = probability_backscatter_change(diff)
    assert float(result.values[0]) > 0.5


def test_backscatter_below_threshold_lt_half():
    diff = xr.DataArray([-5.0], dims=["x"])
    result = probability_backscatter_change(diff)
    assert float(result.values[0]) < 0.5


def test_backscatter_output_in_0_1():
    diff = xr.DataArray([-100.0, 0.0, 100.0], dims=["x"])
    result = probability_backscatter_change(diff)
    assert float(result.values.min()) >= 0.0
    assert float(result.values.max()) <= 1.0


def test_backscatter_output_name():
    diff = xr.DataArray([1.5], dims=["x"])
    result = probability_backscatter_change(diff)
    assert result.name == "p_avalanche"


def test_backscatter_dims_preserved():
    diff = xr.DataArray([[0.0, 1.5], [3.0, 5.0]], dims=["y", "x"])
    result = probability_backscatter_change(diff)
    assert result.dims == diff.dims


# ---------------------------------------------------------------------------
# probability_forest_cover
# ---------------------------------------------------------------------------

def test_forest_at_midpoint_is_half():
    fcf = xr.DataArray([50.0], dims=["x"])
    result = probability_forest_cover(fcf, midpoint=50.0)
    assert float(result.values[0]) == pytest.approx(0.5, abs=1e-4)


def test_forest_zero_coverage_gives_high_prob():
    fcf = xr.DataArray([0.0], dims=["x"])
    result = probability_forest_cover(fcf)
    assert float(result.values[0]) > 0.5


def test_forest_full_coverage_gives_low_prob():
    fcf = xr.DataArray([100.0], dims=["x"])
    result = probability_forest_cover(fcf)
    assert float(result.values[0]) < 0.5


def test_forest_output_in_0_1():
    fcf = xr.DataArray([0.0, 50.0, 100.0], dims=["x"])
    result = probability_forest_cover(fcf)
    assert float(result.values.min()) >= 0.0
    assert float(result.values.max()) <= 1.0


def test_forest_output_name():
    fcf = xr.DataArray([50.0], dims=["x"])
    result = probability_forest_cover(fcf)
    assert result.name == "p_avalanche_forest"


# ---------------------------------------------------------------------------
# probability_cell_counts
# ---------------------------------------------------------------------------

def test_cell_counts_at_midpoint_is_half_with_log():
    """use_log=True: midpoint in log1p space → prob = 0.5."""
    midpoint = 20.0
    counts = xr.DataArray([midpoint], dims=["x"])
    result = probability_cell_counts(counts, midpoint=midpoint, use_log=True)
    assert float(result.values[0]) == pytest.approx(0.5, abs=1e-4)


def test_cell_counts_at_midpoint_is_half_without_log():
    midpoint = 20.0
    counts = xr.DataArray([midpoint], dims=["x"])
    result = probability_cell_counts(counts, midpoint=midpoint, use_log=False)
    assert float(result.values[0]) == pytest.approx(0.5, abs=1e-4)


def test_cell_counts_zero_gives_low_prob():
    counts = xr.DataArray([0.0], dims=["x"])
    result = probability_cell_counts(counts)
    assert float(result.values[0]) < 0.5


def test_cell_counts_high_value_gives_high_prob():
    counts = xr.DataArray([1000.0], dims=["x"])
    result = probability_cell_counts(counts)
    assert float(result.values[0]) > 0.9


def test_cell_counts_output_name():
    counts = xr.DataArray([10.0], dims=["x"])
    result = probability_cell_counts(counts)
    assert result.name == "p_avalanche_cells"


# ---------------------------------------------------------------------------
# probability_slope_angle
# ---------------------------------------------------------------------------

def test_slope_at_midpoint_is_half():
    slope = xr.DataArray([35.0], dims=["x"])
    result = probability_slope_angle(slope, midpoint=35.0)
    assert float(result.values[0]) == pytest.approx(0.5, abs=1e-4)


def test_flat_slope_gives_high_prob():
    slope = xr.DataArray([0.0], dims=["x"])
    result = probability_slope_angle(slope)
    assert float(result.values[0]) > 0.5


def test_steep_slope_gives_low_prob():
    slope = xr.DataArray([80.0], dims=["x"])
    result = probability_slope_angle(slope)
    assert float(result.values[0]) < 0.5


def test_slope_radians_auto_converted():
    """Input in radians (< 4π) should be auto-converted to degrees."""
    slope_deg = xr.DataArray([35.0], dims=["x"])
    slope_rad = xr.DataArray([np.deg2rad(35.0)], dims=["x"])

    result_deg = probability_slope_angle(slope_deg, midpoint=35.0)
    result_rad = probability_slope_angle(slope_rad, midpoint=35.0)

    np.testing.assert_allclose(result_deg.values, result_rad.values, atol=1e-4)


def test_slope_output_name():
    slope = xr.DataArray([35.0], dims=["x"])
    result = probability_slope_angle(slope)
    assert result.name == "p_avalanche_slope"


# ---------------------------------------------------------------------------
# probability_swe_accumulation
# ---------------------------------------------------------------------------

@pytest.fixture
def swe_da():
    """SWE DataArray with snowmodel_time coordinate over 10 daily steps."""
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    data = np.linspace(0, 100, 10)  # Increasing SWE
    return xr.DataArray(
        data,
        dims=["snowmodel_time"],
        coords={"snowmodel_time": times},
    )


def test_swe_missing_coord_raises():
    bad_swe = xr.DataArray([1.0, 2.0], dims=["time"])
    with pytest.raises(ValueError, match="snowmodel_time"):
        probability_swe_accumulation(bad_swe, avalanche_date="2024-01-08")


def test_swe_positive_accumulation_gives_high_prob(swe_da):
    # 7-day accumulation ending 2024-01-08 is positive → prob > 0.5 with midpoint=0
    result = probability_swe_accumulation(swe_da, avalanche_date="2024-01-08", midpoint=0.0)
    assert float(result) > 0.5


def test_swe_zero_accumulation_near_half():
    """Constant SWE → zero accumulation → prob ≈ 0.5 with midpoint=0."""
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    data = np.ones(10) * 50.0  # Constant SWE
    swe = xr.DataArray(data, dims=["snowmodel_time"], coords={"snowmodel_time": times})
    result = probability_swe_accumulation(swe, avalanche_date="2024-01-08", midpoint=0.0)
    assert float(result) == pytest.approx(0.5, abs=0.01)


def test_swe_negative_accumulation_clipped_to_zero(swe_da):
    """SWE melt (negative accumulation) should be clipped to 0, not decrease prob."""
    # Decreasing SWE scenario
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    data = np.linspace(100, 0, 10)  # Decreasing SWE (melt)
    swe = xr.DataArray(data, dims=["snowmodel_time"], coords={"snowmodel_time": times})
    result = probability_swe_accumulation(swe, avalanche_date="2024-01-08", midpoint=0.0)
    # Clipped to 0 accumulation → prob ≈ 0.5
    assert float(result) == pytest.approx(0.5, abs=0.01)


def test_swe_output_attrs_contain_metadata(swe_da):
    result = probability_swe_accumulation(
        swe_da, avalanche_date="2024-01-08", accumulation_days=5
    )
    assert result.attrs["accumulation_days"] == 5
    assert "avalanche_date" in result.attrs


def test_swe_output_name(swe_da):
    result = probability_swe_accumulation(swe_da, avalanche_date="2024-01-08")
    assert result.name == "p_avalanche_swe"
