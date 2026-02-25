import numpy as np
import pytest
import xarray as xr

from sarvalanche.probabilities.z_score import z_score_to_probability


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_at_threshold_prob_is_half():
    distance = xr.DataArray([2.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=2.0)
    assert float(result.values[0]) == pytest.approx(0.5)


def test_above_threshold_prob_gt_half():
    distance = xr.DataArray([4.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=2.0)
    assert float(result.values[0]) > 0.5


def test_below_threshold_prob_lt_half():
    distance = xr.DataArray([0.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=2.0)
    assert float(result.values[0]) < 0.5


def test_large_positive_z_prob_near_one():
    distance = xr.DataArray([20.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=2.0)
    assert float(result.values[0]) > 0.99


def test_large_negative_z_prob_near_zero():
    distance = xr.DataArray([-20.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=2.0)
    assert float(result.values[0]) < 0.01


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_output_name():
    distance = xr.DataArray([2.0], dims=["x"])
    result = z_score_to_probability(distance)
    assert result.name == "ml_probability"


def test_output_units_attr():
    distance = xr.DataArray([2.0], dims=["x"])
    result = z_score_to_probability(distance)
    assert result.attrs["units"] == "probability [0, 1]"


def test_threshold_stored_in_attrs():
    distance = xr.DataArray([2.0], dims=["x"])
    result = z_score_to_probability(distance, threshold=3.0)
    assert result.attrs["threshold"] == 3.0


def test_input_attrs_propagated():
    distance = xr.DataArray([2.0], dims=["x"], attrs={"sensor": "S1"})
    result = z_score_to_probability(distance, threshold=2.0)
    assert result.attrs["sensor"] == "S1"


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

def test_custom_threshold_changes_midpoint():
    distance = xr.DataArray([5.0], dims=["x"])
    r_default = z_score_to_probability(distance, threshold=2.0)
    r_shifted = z_score_to_probability(distance, threshold=5.0)
    # At z=5, threshold=5 → p=0.5; threshold=2 → p>0.5
    assert float(r_default.values[0]) > float(r_shifted.values[0])
    assert float(r_shifted.values[0]) == pytest.approx(0.5)
