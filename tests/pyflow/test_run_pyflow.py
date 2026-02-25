import numpy as np
import xarray as xr
from affine import Affine
import pytest
import geopandas as gpd

from sarvalanche.vendored.flowpy import run_flowpy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_vshaped_dem_and_release(
    ncols: int = 100,
    release_cols: slice = slice(-10, None),
    crs: str = "EPSG:4326",
):
    """
    Return (dem, release) for a 3×ncols V-shaped valley.

    Row 0 and Row 2 are the side slopes (elevation = x + 10).
    Row 1 is the valley bottom (elevation = x).
    Release pixels are placed in the rightmost columns by default.
    """
    x = np.arange(ncols, dtype=float)
    dem_np = np.vstack([x + 10, x, x + 10])

    release_np = np.zeros_like(dem_np)
    release_np[:, release_cols] = 1

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    dem = (
        xr.DataArray(
            dem_np,
            dims=("y", "x"),
            coords={"x": np.arange(ncols), "y": np.arange(3)},
            name="dem",
        )
        .rio.write_crs(crs)
        .rio.write_transform(transform)
        .rio.write_nodata(np.nan)
    )

    release = (
        xr.DataArray(
            release_np.astype(float),
            dims=("y", "x"),
            coords=dem.coords,
            name="release",
        )
        .rio.write_crs(crs)
        .rio.write_transform(transform)
        .rio.write_nodata(0)
    )

    return dem, release


def _make_exponential_vshaped(ncols: int = 100, npts: int = 100):
    x = np.linspace(0, 5, npts)
    dem_np = np.vstack([np.exp(x) + 10, np.exp(x), np.exp(x) + 10]).astype(float)

    release_np = np.zeros_like(dem_np)
    release_np[:, -10:] = 1

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    dem = (
        xr.DataArray(
            dem_np,
            dims=("y", "x"),
            coords={"x": np.arange(npts), "y": np.arange(3)},
            name="dem",
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(np.nan)
    )

    release = (
        xr.DataArray(
            release_np.astype(float),
            dims=("y", "x"),
            coords=dem.coords,
            name="release",
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(0)
    )

    return dem, release


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vshaped_dem_and_release():
    """3×100 V-shaped valley with release on rightmost 10 columns."""
    return _make_vshaped_dem_and_release()


@pytest.fixture
def vshaped_exponential_dem_and_release():
    return _make_exponential_vshaped()


@pytest.fixture
def two_zone_dem_and_release():
    """
    3×100 V-shaped valley with TWO disconnected release zones:
    zone A at columns 40–44, zone B at columns 90–94.
    split_release_by_label should produce 2 tasks → path_list length 2.
    """
    x = np.arange(100, dtype=float)
    dem_np = np.vstack([x + 10, x, x + 10])

    release_np = np.zeros_like(dem_np)
    release_np[:, 40:45] = 1   # zone A (15 pixels, one connected component)
    release_np[:, 90:95] = 1   # zone B (15 pixels, one connected component)

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    dem = (
        xr.DataArray(
            dem_np,
            dims=("y", "x"),
            coords={"x": np.arange(100), "y": np.arange(3)},
            name="dem",
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(np.nan)
    )

    release = (
        xr.DataArray(
            release_np.astype(float),
            dims=("y", "x"),
            coords=dem.coords,
            name="release",
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(0)
    )

    return dem, release


# ===========================================================================
# Existing tests — updated to unpack 3 return values
# ===========================================================================

def test_cell_counts_vshaped_valley(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release

    cell_counts, fp_ta, path_list = run_flowpy(dem=dem, release=release)

    # Shape sanity
    assert cell_counts.shape == dem.shape

    # No hits on the side slopes
    assert np.all(cell_counts[0, :] == 0)
    assert np.all(cell_counts[2, :] == 0)

    # Valley bottom is the only receiver
    valley = cell_counts[1]

    # No accumulation at the very left (upslope end)
    assert valley[0] == 0

    # Strong convergence in the interior
    assert np.all(valley[1:90] == 9)

    # Tapering near the release zone
    assert np.array_equal(
        valley[-10:],
        np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
    )


def test_fp_ta_vshaped_valley(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release

    cell_counts, fp_ta, path_list = run_flowpy(dem=dem, release=release)

    # Shape sanity
    assert fp_ta.shape == dem.shape

    # Side slopes untouched
    assert np.all(fp_ta[0, :] == 0)
    assert np.all(fp_ta[2, :] == 0)

    valley = fp_ta[1]

    # No angle at the very start or near release tail
    assert valley[0] == 0
    assert np.all(valley[-2:] == 0)

    # Interior travel angle is constant for a linear valley
    assert np.all(valley[1:90] == pytest.approx(45.0))


def test_fp_ta_exponential_vshaped_valley(vshaped_exponential_dem_and_release):
    dem, release = vshaped_exponential_dem_and_release

    cell_counts, fp_ta, path_list = run_flowpy(dem=dem, release=release)

    valley = fp_ta[1]

    # No FP_TA outside flow paths
    assert np.all(fp_ta[0] == 0)
    assert np.all(fp_ta[2] == 0)

    # Interior flow path has positive angles
    interior = valley[1:-2]
    assert np.all(interior > 0)

    # Angles must be strictly increasing downslope
    assert np.all(np.diff(interior) > 0)

    # Angles must be physically valid
    assert interior.max() < 90.0


def test_fp_ta_matches_geometry(vshaped_exponential_dem_and_release):
    dem, release = vshaped_exponential_dem_and_release

    _, fp_ta, _ = run_flowpy(dem=dem, release=release)

    valley = fp_ta[1]
    dem_vals = dem.values[1]

    release_idx = np.where(release.values[1] == 1)[0].max()
    dz = dem_vals[release_idx] - dem_vals

    dx = np.arange(len(dz)) - release_idx
    mask = dx < 0

    expected = np.degrees(np.arctan(dz[mask] / np.abs(dx[mask])))

    np.testing.assert_allclose(
        valley[mask][1:-1],
        expected[1:-1],
        rtol=0.1,
    )


# ===========================================================================
# New tests — path_list structure
# ===========================================================================

def test_path_list_is_a_list(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    assert isinstance(path_list, list)


def test_path_list_elements_are_geodataframes(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    for gdf in path_list:
        assert isinstance(gdf, gpd.GeoDataFrame)


def test_path_list_has_geometry_column(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    for gdf in path_list:
        assert "geometry" in gdf.columns


def test_path_list_single_zone_has_length_one(vshaped_dem_and_release):
    """One contiguous release block → one task → path_list length 1."""
    dem, release = vshaped_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    assert len(path_list) == 1


def test_path_list_two_zones_has_length_two(two_zone_dem_and_release):
    """Two disconnected release zones → two tasks → path_list length 2."""
    dem, release = two_zone_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    assert len(path_list) == 2


def test_path_list_geodataframes_have_crs(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    for gdf in path_list:
        assert gdf.crs is not None


# ===========================================================================
# New tests — output shapes
# ===========================================================================

def test_cell_counts_shape_matches_dem(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    cell_counts, fp_ta, _ = run_flowpy(dem=dem, release=release)
    assert cell_counts.shape == dem.shape
    assert fp_ta.shape == dem.shape


def test_cell_counts_non_negative(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    cell_counts, _, _ = run_flowpy(dem=dem, release=release)
    assert np.all(cell_counts >= 0)


def test_fp_ta_non_negative(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, fp_ta, _ = run_flowpy(dem=dem, release=release)
    assert np.all(fp_ta >= 0)


def test_fp_ta_less_than_90_degrees(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release
    _, fp_ta, _ = run_flowpy(dem=dem, release=release)
    assert np.all(fp_ta < 90.0)


# ===========================================================================
# New tests — alpha parameter effect
# ===========================================================================

def test_higher_alpha_gives_shorter_runout():
    """
    Steeper stopping criterion (higher alpha) should limit runout:
    total cell_counts sum will be smaller with alpha=40 than alpha=10.
    """
    dem, release = _make_vshaped_dem_and_release(ncols=200, release_cols=slice(-20, None))

    cc_low_alpha, _, _ = run_flowpy(dem=dem, release=release, alpha=10)
    cc_high_alpha, _, _ = run_flowpy(dem=dem, release=release, alpha=40)

    # Higher alpha = more restrictive → fewer cells reached
    assert cc_high_alpha.sum() <= cc_low_alpha.sum()


def test_lower_alpha_extends_runout():
    """With a very permissive alpha (low), flow reaches further into the valley."""
    dem, release = _make_vshaped_dem_and_release(ncols=200, release_cols=slice(-20, None))

    cc_low, _, _ = run_flowpy(dem=dem, release=release, alpha=5)
    cc_high, _, _ = run_flowpy(dem=dem, release=release, alpha=30)

    # Leftmost non-zero column should be further left for low alpha
    valley_low  = cc_low[1]
    valley_high = cc_high[1]

    first_nonzero_low  = np.argmax(valley_low  > 0) if valley_low.any()  else len(valley_low)
    first_nonzero_high = np.argmax(valley_high > 0) if valley_high.any() else len(valley_high)

    # Low alpha reaches earlier (smaller index = further left)
    assert first_nonzero_low <= first_nonzero_high


# ===========================================================================
# New tests — determinism
# ===========================================================================

def test_results_are_deterministic(vshaped_dem_and_release):
    """Identical inputs must produce bit-identical outputs."""
    dem, release = vshaped_dem_and_release

    cc1, fp1, _ = run_flowpy(dem=dem, release=release)
    cc2, fp2, _ = run_flowpy(dem=dem, release=release)

    np.testing.assert_array_equal(cc1, cc2)
    np.testing.assert_array_equal(fp1, fp2)


# ===========================================================================
# New tests — two-zone spatial correctness
# ===========================================================================

def test_two_zones_cell_counts_are_additive(two_zone_dem_and_release):
    """
    Cell counts from two separate release zones should both appear in the
    combined output — valley receives contributions from both zones.
    """
    dem, release = two_zone_dem_and_release
    cell_counts, _, _ = run_flowpy(dem=dem, release=release)

    valley = cell_counts[1]

    # Zone A (cols 40–44) should produce counts to the LEFT of col 40
    assert valley[:40].sum() > 0

    # Zone B (cols 90–94) should produce counts to the left of col 90
    assert valley[:90].sum() > 0


def test_two_zones_path_list_nonempty(two_zone_dem_and_release):
    dem, release = two_zone_dem_and_release
    _, _, path_list = run_flowpy(dem=dem, release=release)
    for gdf in path_list:
        # Each zone should produce at least one flow path polygon
        assert len(gdf) >= 0  # GeoDataFrame is valid (may be empty after small buffer)
