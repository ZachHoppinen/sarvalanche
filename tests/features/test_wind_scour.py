"""Tests for wind_scour module and wind ridge_method in release mask."""

import numpy as np
import pytest
import xarray as xr

from sarvalanche.features.wind_scour import (
    _sector_mask,
    compute_wind_shelter,
)
from sarvalanche.features.debris_flow_modeling import (
    _cauchy_membership,
    generate_release_mask_simple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_utm_da(data, name="dem"):
    rows, cols = data.shape
    da = xr.DataArray(
        data.astype(float),
        dims=["y", "x"],
        coords={
            "y": np.linspace(6_700_000, 6_700_000 + (rows - 1) * 10, rows),
            "x": np.linspace(500_000,   500_000   + (cols - 1) * 10, cols),
        },
        name=name,
    )
    da = da.rio.write_crs("EPSG:32633")
    da = da.rio.write_transform()
    return da


# ---------------------------------------------------------------------------
# sector_mask
# ---------------------------------------------------------------------------

class TestSectorMask:
    def test_full_circle(self):
        """Tolerance=180 should select all pixels within radius."""
        mask = _sector_mask(5, direction_deg=0, tolerance_deg=180)
        assert mask.shape == (11, 11)
        # Centre pixel is never included
        assert not mask[5, 5]
        # Should have many pixels
        assert mask.sum() > 50

    def test_narrow_sector_north(self):
        """Narrow north-looking sector should only include pixels above centre."""
        mask = _sector_mask(5, direction_deg=0, tolerance_deg=30)
        # Pixels above centre (lower row indices = north) should be selected
        above = mask[:5, :].sum()
        below = mask[6:, :].sum()
        assert above > 0
        assert below == 0

    def test_symmetry(self):
        """Omnidirectional mask should be roughly symmetric."""
        mask = _sector_mask(10, direction_deg=0, tolerance_deg=180)
        assert abs(mask[:10, :].sum() - mask[11:, :].sum()) <= mask.shape[1]


# ---------------------------------------------------------------------------
# compute_wind_shelter
# ---------------------------------------------------------------------------

class TestComputeWindShelter:
    def test_ridge_has_negative_shelter(self):
        """A ridge (narrow, high centre, low sides) should have negative shelter."""
        n = 80
        dem_arr = np.full((n, n), 1000.0)
        # Narrow 2-pixel ridge running north-south — radius must reach
        # the lower terrain on either side.
        dem_arr[:, 39:41] = 1500.0
        dem = _make_utm_da(dem_arr, "dem")

        shelter = compute_wind_shelter(dem, radius_m=80, tolerance_deg=180)
        vals = shelter.values

        # Ridge pixels should have lower (more negative) shelter than
        # the adjacent low terrain (which sees the ridge above it).
        ridge_shelter = np.nanmean(vals[20:60, 39:41])
        low_shelter = np.nanmean(vals[20:60, 30:35])
        assert ridge_shelter < low_shelter, (
            f"Ridge should be more exposed; ridge={ridge_shelter:.2f}, low={low_shelter:.2f}"
        )

    def test_output_shape_and_attrs(self):
        """Output should match input shape and have expected attributes."""
        dem = _make_utm_da(np.random.default_rng(42).uniform(1000, 2000, (30, 30)))
        shelter = compute_wind_shelter(dem, radius_m=30)
        assert shelter.shape == dem.shape
        assert shelter.attrs["product"] == "wind_shelter"
        assert shelter.attrs["units"] == "degrees"

    def test_flat_dem_near_zero(self):
        """A perfectly flat DEM should give shelter values near zero."""
        dem = _make_utm_da(np.full((30, 30), 1500.0))
        shelter = compute_wind_shelter(dem, radius_m=30)
        # All terrain angles are zero on a flat DEM
        assert np.nanmax(np.abs(shelter.values)) < 0.01


# ---------------------------------------------------------------------------
# Wind Cauchy membership
# ---------------------------------------------------------------------------

def test_wind_cauchy_ridge_suppression():
    """Negative shelter (exposed) should give near-zero Cauchy membership."""
    # Default wind Cauchy: a=3, b=10, c=3
    exposed = _cauchy_membership(-5.0, 3, 10, 3)
    sheltered = _cauchy_membership(5.0, 3, 10, 3)
    assert exposed < 0.01, f"Exposed should be near 0, got {exposed}"
    assert sheltered > 0.5, f"Sheltered should be high, got {sheltered}"


# ---------------------------------------------------------------------------
# generate_release_mask_simple with ridge_method="wind"
# ---------------------------------------------------------------------------

def test_wind_method_produces_mask():
    """Wind method should produce a non-empty mask on valid terrain."""
    n = 50
    slope_val = np.deg2rad(40.0)
    slope = _make_utm_da(np.full((n, n), slope_val), "slope")
    fcf = _make_utm_da(np.zeros((n, n)), "fcf")
    # DEM with a valley (low centre, high edges)
    dem_arr = np.full((n, n), 2000.0)
    dem_arr[15:35, 15:35] = 1500.0
    dem = _make_utm_da(dem_arr, "dem")
    fa = _make_utm_da(np.ones((n, n)), "flow_accumulation")

    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=fa,
        forest_cover=fcf, reference=dem,
        ridge_method="wind",
        min_slope_deg=25, fuzzy_threshold=0.15,
        min_release_area_m2=0, smooth=False,
    )
    assert (mask.values > 0).any(), "Wind method should produce non-empty mask"


def test_invalid_ridge_method_raises():
    """Invalid ridge_method should raise ValueError."""
    n = 10
    slope = _make_utm_da(np.full((n, n), np.deg2rad(40)), "slope")
    dem = _make_utm_da(np.full((n, n), 1500), "dem")
    fa = _make_utm_da(np.ones((n, n)), "fa")

    with pytest.raises(ValueError, match="ridge_method"):
        generate_release_mask_simple(
            slope=slope, dem=dem, flow_accum=fa,
            ridge_method="invalid", min_release_area_m2=0, smooth=False,
        )
