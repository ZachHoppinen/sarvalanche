import numpy as np
import xarray as xr
from affine import Affine
import pytest

from sarvalanche.vendored.flowpy import run_flowpy


@pytest.fixture
def vshaped_dem_and_release():
    """
    3x100 V-shaped valley with releases on the rightmost 10 columns.
    """

    x = np.arange(100)

    dem_np = np.vstack(
        [
            x + 10,  # left slope
            x,       # valley bottom
            x + 10,  # right slope
        ]
    ).astype(float)

    release_np = np.zeros_like(dem_np)
    release_np[:, -10:] = 1

    dem = xr.DataArray(
        dem_np,
        dims=("y", "x"),
        coords={
            "x": np.arange(dem_np.shape[1]),
            "y": np.arange(dem_np.shape[0]),
        },
        name="dem",
    )

    release = xr.DataArray(
        release_np,
        dims=("y", "x"),
        coords=dem.coords,
        name="release",
    )

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    dem = (
        dem
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(np.nan)
    )

    release = (
        release
        .astype(float)
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(0)
    )

    return dem, release

def test_cell_counts_vshaped_valley(vshaped_dem_and_release):
    dem, release = vshaped_dem_and_release

    cell_counts, fp_ta = run_flowpy(dem=dem, release=release)

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

    cell_counts, fp_ta = run_flowpy(dem=dem, release=release)

    # Shape sanity
    assert fp_ta.shape == dem.shape

    # Side slopes untouched
    assert np.all(fp_ta[0, :] == 0)
    assert np.all(fp_ta[2, :] == 0)

    valley = fp_ta[1]

    # No angle at the very start or near release tail
    assert valley[0] == 0
    assert np.all(valley[-2:] == 0)

    # Interior travel angle is constant
    assert np.all(valley[1:90] == pytest.approx(45.0))

@pytest.fixture
def vshaped_exponential_dem_and_release():
    x = np.linspace(0, 5, 100)

    dem_np = np.vstack(
        [
            np.exp(x) + 10,
            np.exp(x),
            np.exp(x) + 10,
        ]
    ).astype(float)

    release_np = np.zeros_like(dem_np)
    release_np[:, -10:] = 1

    dem = xr.DataArray(
        dem_np,
        dims=("y", "x"),
        coords={
            "x": np.arange(dem_np.shape[1]),
            "y": np.arange(dem_np.shape[0]),
        },
        name="dem",
    )

    release = xr.DataArray(
        release_np,
        dims=("y", "x"),
        coords=dem.coords,
        name="release",
    )

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    dem = (
        dem
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(np.nan)
    )

    release = (
        release
        .astype(float)
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(transform)
        .rio.write_nodata(0)
    )

    return dem, release

def test_fp_ta_exponential_vshaped_valley(vshaped_exponential_dem_and_release):
    dem, release = vshaped_exponential_dem_and_release

    cell_counts, fp_ta = run_flowpy(dem=dem, release=release)

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

    _, fp_ta = run_flowpy(dem=dem, release=release)

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
