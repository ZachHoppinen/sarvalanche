"""
Pytest suite for SAR detection weight validators.

Run with: pytest test_weight_validators.py -v
"""

import pytest
import numpy as np
import xarray as xr
from sarvalanche.utils.validation import (
    validate_weights_sum_to_one,
    validate_time_weights,
    validate_orbit_weights
)


class TestValidWeights:
    """Tests for valid weight configurations."""

    def test_valid_1d_time_weights(self):
        """Valid 1D time weights should pass validation."""
        weights = xr.DataArray([0.3, 0.7], dims=['time'], coords={'time': [0, 1]})
        assert validate_time_weights(weights) is True

    def test_valid_1d_time_weights_three_timesteps(self):
        """Valid 1D time weights with three timesteps."""
        weights = xr.DataArray([0.2, 0.3, 0.5], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_valid_orbit_weights(self):
        """Valid orbit weights should pass validation."""
        orbit_weights = xr.DataArray(
            [0.4, 0.6],
            dims=['static_orbit'],
            coords={'static_orbit': ['ascending', 'descending']}
        )
        assert validate_orbit_weights(orbit_weights) is True

    def test_valid_2d_time_weights(self):
        """Valid 2D array with time weights at multiple locations."""
        weights_2d = xr.DataArray(
            [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]],
            dims=['location', 'time'],
            coords={'location': [0, 1, 2], 'time': [0, 1]}
        )
        assert validate_time_weights(weights_2d) is True

    def test_valid_3d_spatial_temporal_weights(self):
        """Valid 3D array with time weights at spatial grid points."""
        weights_3d = xr.DataArray(
            np.array([
                [[0.25, 0.75], [0.3, 0.7]],
                [[0.4, 0.6], [0.5, 0.5]]
            ]),
            dims=['y', 'x', 'time'],
            coords={'y': [0, 1], 'x': [0, 1], 'time': [0, 1]}
        )
        assert validate_time_weights(weights_3d) is True

    def test_equal_weights(self):
        """Equal weights should pass validation."""
        weights = xr.DataArray([0.25, 0.25, 0.25, 0.25], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_weights_within_tolerance(self):
        """Weights that sum to 1.0 within tolerance should pass."""
        # Sum is 1.0000001 (within default tolerance of 1e-6)
        weights = xr.DataArray([0.33333334, 0.33333333, 0.33333333], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_single_weight(self):
        """Single weight of 1.0 should pass."""
        weights = xr.DataArray([1.0], dims=['time'])
        assert validate_time_weights(weights) is True


class TestInvalidWeights:
    """Tests for invalid weight configurations that should raise errors."""

    def test_invalid_sum_too_low(self):
        """Weights that sum to less than 1.0 should fail."""
        weights = xr.DataArray([0.3, 0.6], dims=['time'])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights)

    def test_invalid_sum_too_high(self):
        """Weights that sum to more than 1.0 should fail."""
        weights = xr.DataArray([0.6, 0.7], dims=['time'])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights)

    def test_negative_weights(self):
        """Negative weights should fail."""
        weights = xr.DataArray([0.5, -0.5, 1.0], dims=['time'])
        with pytest.raises(ValueError, match="negative"):
            validate_time_weights(weights)

    def test_nan_values(self):
        """NaN values should fail."""
        weights = xr.DataArray([0.3, np.nan, 0.7], dims=['time'])
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_time_weights(weights)

    def test_inf_values(self):
        """Infinite values should fail."""
        weights = xr.DataArray([0.3, np.inf, 0.7], dims=['time'])
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_time_weights(weights)

    def test_all_zeros(self):
        """All zero weights should fail."""
        weights = xr.DataArray([0.0, 0.0, 0.0], dims=['time'])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights)

    def test_2d_one_invalid_location(self):
        """2D array where one location has invalid weights should fail."""
        weights_2d = xr.DataArray(
            [[0.3, 0.7], [0.4, 0.5], [0.5, 0.5]],  # Second location sums to 0.9
            dims=['location', 'time'],
            coords={'location': [0, 1, 2], 'time': [0, 1]}
        )
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights_2d)

    def test_outside_tolerance(self):
        """Weights outside tolerance should fail."""
        # Sum is 1.000002 (2e-6, outside default tolerance of 1e-6)
        weights = xr.DataArray([0.333335, 0.333334, 0.333333], dims=['time'])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights)
    
    def test_outside_tolerance_more_obvious(self):
        """Weights clearly outside tolerance should fail."""
        # Sum is 1.00001 (1e-5, well outside default tolerance of 1e-6)
        weights = xr.DataArray([0.5, 0.5, 0.00001], dims=['time'])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_time_weights(weights)

class TestDimensionHandling:
    """Tests for dimension detection and handling."""

    def test_auto_detect_time_dimension(self):
        """Should auto-detect 'time' dimension."""
        weights = xr.DataArray([0.25, 0.75], dims=['time'])
        assert validate_weights_sum_to_one(weights) is True

    def test_auto_detect_static_orbit_dimension(self):
        """Should auto-detect 'static_orbit' dimension."""
        weights = xr.DataArray([0.4, 0.6], dims=['static_orbit'])
        assert validate_weights_sum_to_one(weights) is True

    def test_explicit_dimension_specification(self):
        """Should use explicitly specified dimension."""
        weights = xr.DataArray([0.3, 0.7], dims=['custom_dim'])
        assert validate_weights_sum_to_one(weights, dim='custom_dim') is True

    def test_missing_time_and_orbit_dimensions(self):
        """Should raise error if neither time nor static_orbit found."""
        weights = xr.DataArray([0.3, 0.7], dims=['other_dim'])
        with pytest.raises(KeyError, match="Neither 'time' nor 'static_orbit'"):
            validate_weights_sum_to_one(weights)

    def test_nonexistent_dimension(self):
        """Should raise error if specified dimension doesn't exist."""
        weights = xr.DataArray([0.3, 0.7], dims=['time'])
        with pytest.raises(KeyError, match="Dimension 'nonexistent' not found"):
            validate_weights_sum_to_one(weights, dim='nonexistent')

    def test_prefer_time_over_orbit(self):
        """When both dimensions exist, should prefer 'time' if not specified."""
        # Create array with both dimensions
        weights = xr.DataArray(
            [[0.3, 0.7], [0.4, 0.6]],
            dims=['static_orbit', 'time']
        )
        # Should validate along 'time' by default
        assert validate_weights_sum_to_one(weights) is True


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_validate_time_weights_function(self):
        """validate_time_weights should work correctly."""
        weights = xr.DataArray([0.4, 0.6], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_validate_orbit_weights_function(self):
        """validate_orbit_weights should work correctly."""
        weights = xr.DataArray([0.55, 0.45], dims=['static_orbit'])
        assert validate_orbit_weights(weights) is True

    def test_time_weights_wrong_dimension(self):
        """validate_time_weights should fail if 'time' dimension missing."""
        weights = xr.DataArray([0.4, 0.6], dims=['static_orbit'])
        with pytest.raises(KeyError, match="'time' not found"):
            validate_time_weights(weights)

    def test_orbit_weights_wrong_dimension(self):
        """validate_orbit_weights should fail if 'static_orbit' dimension missing."""
        weights = xr.DataArray([0.4, 0.6], dims=['time'])
        with pytest.raises(KeyError, match="'static_orbit' not found"):
            validate_orbit_weights(weights)


class TestCustomTolerance:
    """Tests for custom tolerance settings."""

    def test_strict_tolerance(self):
        """Stricter tolerance should catch smaller deviations."""
        weights = xr.DataArray([0.4, 0.3, 0.3000001], dims=['time'])
        # Should pass with default tolerance (1e-6)
        assert validate_time_weights(weights, tolerance=1e-6) is True
        # Should fail with stricter tolerance
        with pytest.raises(ValueError):
            validate_time_weights(weights, tolerance=1e-8)

    def test_loose_tolerance(self):
        """Looser tolerance should allow larger deviations."""
        weights = xr.DataArray([0.45, 0.54], dims=['time'])  # Sum = 0.99
        # Should fail with default tolerance
        with pytest.raises(ValueError):
            validate_time_weights(weights, tolerance=1e-6)
        # Should pass with looser tolerance
        assert validate_time_weights(weights, tolerance=0.02) is True


class TestRaiseOnFailParameter:
    """Tests for raise_on_fail parameter behavior."""

    def test_raise_on_fail_true(self):
        """When raise_on_fail=True, should raise ValueError."""
        weights = xr.DataArray([0.3, 0.6], dims=['time'])
        with pytest.raises(ValueError):
            validate_weights_sum_to_one(weights, raise_on_fail=True)

    def test_raise_on_fail_false(self):
        """When raise_on_fail=False, should return False and warn."""
        weights = xr.DataArray([0.3, 0.6], dims=['time'])
        with pytest.warns(UserWarning, match="do not sum to 1.0"):
            result = validate_weights_sum_to_one(weights, raise_on_fail=False)
        assert result is False

    def test_raise_on_fail_false_negative_weights(self):
        """Should warn about negative weights when raise_on_fail=False."""
        weights = xr.DataArray([1.5, -0.5], dims=['time'])
        with pytest.warns(UserWarning, match="negative"):
            result = validate_weights_sum_to_one(weights, raise_on_fail=False)
        assert result is False

    def test_raise_on_fail_false_nan_weights(self):
        """Should warn about NaN when raise_on_fail=False."""
        weights = xr.DataArray([0.5, np.nan], dims=['time'])
        with pytest.warns(UserWarning, match="NaN or infinite"):
            result = validate_weights_sum_to_one(weights, raise_on_fail=False)
        assert result is False


class TestErrorMessages:
    """Tests for quality and informativeness of error messages."""

    def test_error_message_includes_statistics(self):
        """Error message should include min/max/mean statistics."""
        weights = xr.DataArray([0.3, 0.6], dims=['time'])
        with pytest.raises(ValueError) as exc_info:
            validate_time_weights(weights)

        error_msg = str(exc_info.value)
        assert "min=" in error_msg
        assert "max=" in error_msg
        assert "mean=" in error_msg

    def test_error_message_includes_dimension_name(self):
        """Error message should mention the dimension being validated."""
        weights = xr.DataArray([0.3, 0.6], dims=['time'])
        with pytest.raises(ValueError) as exc_info:
            validate_time_weights(weights)

        assert "dimension 'time'" in str(exc_info.value)

    def test_error_message_includes_worst_cases(self):
        """Error message should show worst-case violations."""
        weights_2d = xr.DataArray(
            [[0.3, 0.7], [0.2, 0.7], [0.5, 0.5]],  # Second location sums to 0.9
            dims=['location', 'time'],
            coords={'location': [10, 20, 30], 'time': [0, 1]}
        )
        with pytest.raises(ValueError) as exc_info:
            validate_time_weights(weights_2d)

        error_msg = str(exc_info.value)
        assert "Worst cases:" in error_msg
        assert "sum=" in error_msg


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_weights(self):
        """Very small but valid weights should pass."""
        weights = xr.DataArray([0.001, 0.999], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_many_weights(self):
        """Large number of weights should be handled correctly."""
        n = 100
        weights = xr.DataArray(np.ones(n) / n, dims=['time'])
        assert validate_time_weights(weights) is True

    def test_high_dimensional_array(self):
        """High-dimensional arrays should work correctly."""
        # 4D array: (y, x, orbit, time)
        weights_4d = xr.DataArray(
            np.random.dirichlet([1, 1, 1], size=(2, 3, 2)),  # Generates valid weights
            dims=['y', 'x', 'orbit', 'time']
        )
        # Reshape to ensure proper dimensions
        weights_4d = xr.DataArray(
            np.array([[[[0.3, 0.7], [0.4, 0.6]],
                       [[0.5, 0.5], [0.2, 0.8]],
                       [[0.35, 0.65], [0.45, 0.55]]]]),
            dims=['y', 'x', 'orbit', 'time']
        )
        assert validate_time_weights(weights_4d) is True

    def test_single_element_array(self):
        """Single element array with weight 1.0 should pass."""
        weights = xr.DataArray([1.0], dims=['time'])
        assert validate_time_weights(weights) is True

    def test_float32_dtype(self):
        """Should work with float32 dtype."""
        weights = xr.DataArray(
            np.array([0.3, 0.7], dtype=np.float32),
            dims=['time']
        )
        assert validate_time_weights(weights) is True

    def test_float64_dtype(self):
        """Should work with float64 dtype."""
        weights = xr.DataArray(
            np.array([0.4, 0.6], dtype=np.float64),
            dims=['time']
        )
        assert validate_time_weights(weights) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])