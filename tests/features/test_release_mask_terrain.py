"""
Integration tests for generate_release_mask_simple in debris_flow_modeling.

All inputs are synthetic 30×30 UTM arrays (10 m pixels), so no file I/O needed.
"""
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import label as ndlabel

from sarvalanche.features.debris_flow_modeling import generate_release_mask_simple


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

    slope = 35° (0.611 rad), forest = 0,
    DEM = simple slope from 600 to 300 m (top-to-bottom),
    flow_accum = 1 everywhere (no channels).
    """
    n = 30
    slope_val = np.deg2rad(35.0)
    slope_arr = np.full((n, n), slope_val)
    fcf_arr   = np.zeros((n, n))
    dem_arr   = np.linspace(600, 300, n)[:, None] * np.ones((1, n))
    fa_arr    = np.ones((n, n))

    slope      = _make_utm_da(slope_arr, "slope")
    fcf        = _make_utm_da(fcf_arr,   "fcf")
    dem        = _make_utm_da(dem_arr,    "dem")
    flow_accum = _make_utm_da(fa_arr,     "flow_accumulation")
    return slope, fcf, dem, flow_accum


# Shared kwargs to disable size filter and smoothing for unit-test clarity
_COMMON = dict(
    min_slope_deg=25,
    fuzzy_threshold=0.15,
    min_release_area_m2=0,
    smooth=False,
)


# ---------------------------------------------------------------------------
# Basic: slope + FCF filter produces a non-empty mask
# ---------------------------------------------------------------------------

def test_basic_mask_nonempty(base_inputs):
    """All valid pixels should survive slope+FCF filter."""
    slope, fcf, dem, flow_accum = base_inputs
    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=flow_accum,
        forest_cover=fcf, reference=dem, **_COMMON,
    )
    assert (mask.values > 0).any(), "Expected non-empty mask"


# ---------------------------------------------------------------------------
# Flow accumulation filter: entire top half is a channel
# ---------------------------------------------------------------------------

def test_flow_accum_filter_reduces_mask(base_inputs):
    """High flow_accum pixels reduce the total mask area."""
    slope, fcf, dem, _ = base_inputs
    n = 30

    # Baseline: low flow accum everywhere
    fa_low = _make_utm_da(np.ones((n, n)), "flow_accumulation")
    mask_all = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=fa_low,
        forest_cover=fcf, reference=dem, max_flow_accum_channel=10.0,
        **_COMMON,
    )

    # High flow accum in top half
    fa_arr = np.ones((n, n))
    fa_arr[0:15, :] = 2000.0
    fa_high = _make_utm_da(fa_arr, "flow_accumulation")
    mask_filtered = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=fa_high,
        forest_cover=fcf, reference=dem, max_flow_accum_channel=10.0,
        **_COMMON,
    )

    assert mask_filtered.values.sum() < mask_all.values.sum(), (
        "High flow_accum should reduce the mask area"
    )


# ---------------------------------------------------------------------------
# Forest cover filter: top half is forested
# ---------------------------------------------------------------------------

def test_fcf_filter_reduces_mask(base_inputs):
    """High forest cover pixels reduce the total mask area."""
    slope, _, dem, flow_accum = base_inputs
    n = 30

    # Baseline: no forest
    fcf_none = _make_utm_da(np.zeros((n, n)), "fcf")
    mask_all = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=flow_accum,
        forest_cover=fcf_none, reference=dem, **_COMMON,
    )

    # Dense forest in top half (FCF=95 pushes fuzzy score below threshold)
    fcf_arr = np.zeros((n, n))
    fcf_arr[0:15, :] = 95.0
    fcf_forest = _make_utm_da(fcf_arr, "fcf")
    mask_filtered = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=flow_accum,
        forest_cover=fcf_forest, reference=dem, **_COMMON,
    )

    assert mask_filtered.values.sum() < mask_all.values.sum(), (
        "Forested area should reduce the mask"
    )


# ---------------------------------------------------------------------------
# Slope filter: only a band of valid slope
# ---------------------------------------------------------------------------

def test_slope_filter_flat_produces_empty():
    """Entirely flat slope should produce an empty mask."""
    n = 30
    slope = _make_utm_da(np.full((n, n), np.deg2rad(10.0)), "slope")
    dem   = _make_utm_da(np.linspace(600, 300, n)[:, None] * np.ones((1, n)), "dem")
    fa    = _make_utm_da(np.ones((n, n)), "flow_accumulation")
    fcf   = _make_utm_da(np.zeros((n, n)), "fcf")

    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=fa,
        forest_cover=fcf, reference=dem, **_COMMON,
    )
    assert (mask.values == 0).all(), "Entirely flat region should produce empty mask"


def test_slope_filter_steep_produces_empty():
    """Entirely over-steep slope should produce an empty mask."""
    n = 30
    slope = _make_utm_da(np.full((n, n), np.deg2rad(70.0)), "slope")
    dem   = _make_utm_da(np.linspace(600, 300, n)[:, None] * np.ones((1, n)), "dem")
    fa    = _make_utm_da(np.ones((n, n)), "flow_accumulation")
    fcf   = _make_utm_da(np.zeros((n, n)), "fcf")

    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=fa,
        forest_cover=fcf, reference=dem, **_COMMON,
    )
    assert (mask.values == 0).all(), "Entirely over-steep region should produce empty mask"


# ---------------------------------------------------------------------------
# Size filter
# ---------------------------------------------------------------------------

def test_size_filter_removes_small_zones(base_inputs):
    """Zones smaller than min_release_area_m2 are removed."""
    _, fcf, dem, flow_accum = base_inputs
    n = 30

    # Only a tiny 2x2 patch of valid slope, rest flat
    slope_arr = np.full((n, n), np.deg2rad(10.0))
    slope_arr[14:16, 14:16] = np.deg2rad(35.0)
    slope = _make_utm_da(slope_arr, "slope")

    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=flow_accum,
        forest_cover=fcf, reference=dem,
        min_slope_deg=25, fuzzy_threshold=0.15,
        min_release_area_m2=10000,  # 100x100m — much larger than 2x2 at 10m
        smooth=False,
    )
    assert (mask.values == 0).all(), "Tiny zone should be removed by size filter"


# ---------------------------------------------------------------------------
# No forest cover (optional param)
# ---------------------------------------------------------------------------

def test_no_forest_cover(base_inputs):
    """forest_cover=None should work without error."""
    slope, _, dem, flow_accum = base_inputs

    mask = generate_release_mask_simple(
        slope=slope, dem=dem, flow_accum=flow_accum,
        forest_cover=None, reference=dem, **_COMMON,
    )
    assert (mask.values > 0).any(), "Should produce non-empty mask without FCF"
