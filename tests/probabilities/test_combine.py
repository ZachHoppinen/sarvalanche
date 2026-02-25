import numpy as np
import pytest
import xarray as xr

from sarvalanche.probabilities.combine import combine_probabilities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_probs():
    """Two probability values along track_pol dim."""
    return xr.DataArray([0.2, 0.8], dims=["track_pol"])


@pytest.fixture
def equal_weights():
    return xr.DataArray([0.5, 0.5], dims=["track_pol"])


# ---------------------------------------------------------------------------
# weighted_mean
# ---------------------------------------------------------------------------

def test_weighted_mean_equal_weights(two_probs, equal_weights):
    result = combine_probabilities(two_probs, equal_weights, dim="track_pol", method="weighted_mean")
    assert float(result) == pytest.approx(0.5)


def test_weighted_mean_unequal_weights():
    probs = xr.DataArray([0.0, 1.0], dims=["track_pol"])
    weights = xr.DataArray([0.25, 0.75], dims=["track_pol"])
    result = combine_probabilities(probs, weights, dim="track_pol", method="weighted_mean")
    assert float(result) == pytest.approx(0.75)


def test_weighted_mean_uniform_weights_by_default(two_probs):
    result = combine_probabilities(two_probs, dim="track_pol", method="weighted_mean")
    assert float(result) == pytest.approx(0.5)


def test_weighted_mean_output_in_0_1_range():
    probs = xr.DataArray([0.0, 0.5, 1.0], dims=["track_pol"])
    result = combine_probabilities(probs, dim="track_pol", method="weighted_mean")
    assert 0.0 <= float(result) <= 1.0


# ---------------------------------------------------------------------------
# log_odds
# ---------------------------------------------------------------------------

def test_log_odds_symmetric_probs_returns_half():
    # log-odds(0.2) and log-odds(0.8) are symmetric around 0; mean → 0 → p=0.5
    probs = xr.DataArray([0.2, 0.8], dims=["track_pol"])
    weights = xr.DataArray([0.5, 0.5], dims=["track_pol"])
    result = combine_probabilities(probs, weights, dim="track_pol", method="log_odds")
    assert float(result) == pytest.approx(0.5, abs=1e-4)


def test_log_odds_high_probs_stay_high():
    probs = xr.DataArray([0.9, 0.95], dims=["track_pol"])
    result = combine_probabilities(probs, dim="track_pol", method="log_odds")
    assert float(result) > 0.9


def test_log_odds_output_in_0_1_range():
    probs = xr.DataArray([0.01, 0.99], dims=["track_pol"])
    result = combine_probabilities(probs, dim="track_pol", method="log_odds")
    assert 0.0 <= float(result) <= 1.0


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

def test_invalid_method_raises(two_probs, equal_weights):
    with pytest.raises(ValueError, match="Unknown method"):
        combine_probabilities(two_probs, equal_weights, dim="track_pol", method="bad_method")


# ---------------------------------------------------------------------------
# agreement_boosting
# ---------------------------------------------------------------------------

def test_agreement_boosting_increases_combined_probability():
    # Both sources detect something (prob > min_prob_threshold)
    probs = xr.DataArray([0.6, 0.7], dims=["track_pol"])
    weights = xr.DataArray([0.5, 0.5], dims=["track_pol"])

    plain = combine_probabilities(probs, weights, dim="track_pol", agreement_boosting=False)
    boosted = combine_probabilities(
        probs, weights, dim="track_pol",
        agreement_boosting=True, agreement_strength=0.8
    )
    assert float(boosted) > float(plain)


def test_agreement_boosting_no_boost_when_one_source_below_threshold():
    # One source below min_prob_threshold → agreement fraction = 0.5
    probs = xr.DataArray([0.05, 0.7], dims=["track_pol"])
    weights = xr.DataArray([0.5, 0.5], dims=["track_pol"])

    plain = combine_probabilities(probs, weights, dim="track_pol", agreement_boosting=False)
    boosted = combine_probabilities(
        probs, weights, dim="track_pol",
        agreement_boosting=True, min_prob_threshold=0.1, agreement_strength=0.8
    )
    # Boost is only partial (agreement=0.5) so result still ≥ plain
    assert float(boosted) >= float(plain)


def test_agreement_boosting_all_below_threshold_returns_zero():
    """All sources below threshold → no valid sources → output 0."""
    probs = xr.DataArray([0.01, 0.02], dims=["track_pol"])
    result = combine_probabilities(
        probs, dim="track_pol",
        agreement_boosting=True, min_prob_threshold=0.1
    )
    assert float(result) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Output is always clipped to [0, 1]
# ---------------------------------------------------------------------------

def test_output_clipped_to_1():
    # Force high probabilities
    probs = xr.DataArray([1.0, 1.0], dims=["track_pol"])
    result = combine_probabilities(
        probs, dim="track_pol",
        agreement_boosting=True, agreement_strength=1.0
    )
    assert float(result) <= 1.0
