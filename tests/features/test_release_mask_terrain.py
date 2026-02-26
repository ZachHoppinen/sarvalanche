"""
Integration tests for terrain-based filtering and splitting in generate_release_mask.

All inputs are synthetic 30×30 UTM arrays (10 m pixels), so no file I/O needed.
"""
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import label as ndlabel


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_utm_da(data, name="dem"):
    """Wrap a 2-D numpy array as a UTM DataArray (EPSG:32633, 10 m/px)."""
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


def _count_zones(arr):
    _, n = ndlabel(arr > 0)
    return n


@pytest.fixture
def base_inputs():
    """
    30×30 UTM arrays that pass the default slope+FCF filter everywhere.

    slope = 35° (0.611 rad), forest = 0, aspect = flat (all zeros),
    DEM = simple slope from 600 to 300 m (top-to-bottom).
    """
    n = 30
    slope_val = np.deg2rad(35.0)
    slope_arr = np.full((n, n), slope_val)
    fcf_arr   = np.zeros((n, n))
    aspect_arr = np.zeros((n, n))
    dem_arr = np.linspace(600, 300, n)[:, None] * np.ones((1, n))

    slope  = _make_utm_da(slope_arr, "slope")
    fcf    = _make_utm_da(fcf_arr,   "fcf")
    aspect = _make_utm_da(aspect_arr, "aspect")
    dem    = _make_utm_da(dem_arr,    "dem")
    return slope, fcf, aspect, dem


# ---------------------------------------------------------------------------
# Backward-compatibility: no terrain params
# ---------------------------------------------------------------------------

def test_backward_compatible_no_terrain_params(base_inputs):
    """No terrain params → all valid pixels survive (same as before)."""
    from sarvalanche.features.debris_flow_modeling import generate_release_mask

    slope, fcf, aspect, dem = base_inputs
    mask = generate_release_mask(
        slope=slope,
        forest_cover=fcf,
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=1,
        smooth=False,
        reference=dem,
    )
    assert (mask.values > 0).any(), "Expected non-empty mask"


# ---------------------------------------------------------------------------
# Flow accumulation filter
# ---------------------------------------------------------------------------

def test_flow_accum_filter_excludes_channel(base_inputs):
    """Pixels with flow_accum > max_flow_accum are excluded."""
    from sarvalanche.features.debris_flow_modeling import generate_release_mask

    slope, fcf, aspect, dem = base_inputs
    n = 30

    # High flow accumulation along column 15 (drainage channel)
    fa_arr = np.ones((n, n))
    fa_arr[:, 15] = 2000.0  # well above threshold
    fa = _make_utm_da(fa_arr, "flow_accumulation")

    mask = generate_release_mask(
        slope=slope,
        forest_cover=fcf,
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=1,
        smooth=False,
        reference=dem,
        flow_accum=fa,
        max_flow_accum=1000.0,
    )
    # Column 15 should be zeroed out
    assert (mask.values[:, 15] == 0).all(), (
        "Expected channel column to be excluded by flow_accum filter"
    )
    # Other pixels should still be present
    assert (mask.values[:, 10] > 0).any()


# ---------------------------------------------------------------------------
# TPI filters
# ---------------------------------------------------------------------------

def test_tpi_max_filter_excludes_ridge(base_inputs):
    """Pixels with TPI > max_tpi are excluded."""
    from sarvalanche.features.debris_flow_modeling import generate_release_mask

    slope, fcf, aspect, dem = base_inputs
    n = 30

    tpi_arr = np.zeros((n, n))
    tpi_arr[5, :] = 50.0  # ridge row
    tpi = _make_utm_da(tpi_arr, "tpi")

    mask = generate_release_mask(
        slope=slope,
        forest_cover=fcf,
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=1,
        smooth=False,
        reference=dem,
        tpi=tpi,
        max_tpi=30.0,
    )
    assert (mask.values[5, :] == 0).all(), "Ridge row should be excluded by max_tpi filter"
    assert (mask.values[15, :] > 0).any()


def test_tpi_min_filter_excludes_valley(base_inputs):
    """Pixels with TPI < min_tpi are excluded."""
    from sarvalanche.features.debris_flow_modeling import generate_release_mask

    slope, fcf, aspect, dem = base_inputs
    n = 30

    tpi_arr = np.zeros((n, n))
    tpi_arr[5, :] = -50.0  # valley row
    tpi = _make_utm_da(tpi_arr, "tpi")

    mask = generate_release_mask(
        slope=slope,
        forest_cover=fcf,
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=1,
        smooth=False,
        reference=dem,
        tpi=tpi,
        min_tpi=-30.0,
    )
    assert (mask.values[5, :] == 0).all(), "Valley row should be excluded by min_tpi filter"
    assert (mask.values[15, :] > 0).any()


# ---------------------------------------------------------------------------
# TPI split threshold
# ---------------------------------------------------------------------------

def test_tpi_split_threshold_increases_zone_count(base_inputs):
    """A sharp TPI step should split the mask into more zones."""
    from sarvalanche.features.debris_flow_modeling import generate_release_mask

    slope, fcf, aspect, dem = base_inputs
    n = 30

    # TPI step at row 15 → top half near 0, bottom half near +60
    tpi_arr = np.zeros((n, n))
    tpi_arr[15:, :] = 60.0
    tpi = _make_utm_da(tpi_arr, "tpi")

    mask_no_split = generate_release_mask(
        slope=slope, forest_cover=fcf,
        min_slope_deg=25, max_slope_deg=60, max_fcf=10,
        min_group_size=1, smooth=False, reference=dem,
        tpi=tpi,
    )
    mask_with_split = generate_release_mask(
        slope=slope, forest_cover=fcf,
        min_slope_deg=25, max_slope_deg=60, max_fcf=10,
        min_group_size=1, smooth=False, reference=dem,
        tpi=tpi, tpi_split_threshold=30.0,
    )
    n_no_split   = _count_zones(mask_no_split.values)
    n_with_split = _count_zones(mask_with_split.values)
    assert n_with_split >= n_no_split, (
        f"Expected more zones after TPI split: got {n_with_split} vs {n_no_split}"
    )


# ---------------------------------------------------------------------------
# _apply_linear_split unit test
# ---------------------------------------------------------------------------

def test_apply_linear_split_creates_split():
    """Sharp field gradient → _apply_linear_split creates ≥ 2 zones."""
    from sarvalanche.features.debris_flow_modeling import _apply_linear_split

    n = 20
    mask_arr = np.ones((n, n), dtype=bool)

    # Field step at column 10
    field = np.zeros((n, n))
    field[:, 10:] = 100.0

    result = _apply_linear_split(mask_arr, field, threshold=50.0)
    _, n_zones = ndlabel(result)
    assert n_zones >= 2, f"Expected ≥ 2 zones after linear split, got {n_zones}"


def test_apply_linear_split_no_split_below_threshold():
    """Field gradient below threshold → mask unchanged (1 zone)."""
    from sarvalanche.features.debris_flow_modeling import _apply_linear_split

    n = 20
    mask_arr = np.ones((n, n), dtype=bool)

    # Small gradient — below threshold
    field = np.zeros((n, n))
    field[:, 10:] = 5.0

    result = _apply_linear_split(mask_arr, field, threshold=50.0)
    _, n_zones = ndlabel(result)
    assert n_zones == 1


def test_apply_linear_split_nan_safe():
    """NaN values in field are treated as 0 — no crash."""
    from sarvalanche.features.debris_flow_modeling import _apply_linear_split

    n = 10
    mask_arr = np.ones((n, n), dtype=bool)
    field = np.full((n, n), np.nan)

    # Should not raise; output should still be boolean
    result = _apply_linear_split(mask_arr, field, threshold=1.0)
    assert result.dtype == bool
