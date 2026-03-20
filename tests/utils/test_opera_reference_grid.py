"""Tests for make_opera_reference_grid and find_utm_crs."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from sarvalanche.utils.constants import OPERA_RESOLUTION
from sarvalanche.utils.grid import make_opera_reference_grid
from sarvalanche.utils.projections import find_utm_crs


# ---------------------------------------------------------------------------
# find_utm_crs
# ---------------------------------------------------------------------------


class TestFindUtmCrs:
    """Tests for find_utm_crs."""

    def test_wgs84_input_returns_correct_utm_zone(self):
        """AOI in WGS84 near Salt Lake City → UTM 12N."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        crs = find_utm_crs(aoi, CRS.from_epsg(4326))
        assert crs == CRS.from_epsg(32612)

    def test_utm_input_returns_same_zone(self):
        """AOI already in UTM 12N should still return UTM 12N."""
        aoi = box(500_000, 4_500_000, 510_000, 4_510_000)
        crs = find_utm_crs(aoi, CRS.from_epsg(32612))
        assert crs == CRS.from_epsg(32612)

    def test_wgs84_western_hemisphere(self):
        """AOI near Anchorage, AK → UTM 6N."""
        aoi = box(-150.0, 61.0, -149.5, 61.5)
        crs = find_utm_crs(aoi, CRS.from_epsg(4326))
        assert crs == CRS.from_epsg(32606)

    def test_different_projected_crs_input(self):
        """AOI in UTM 11N but centroid is well within zone 11."""
        aoi = box(500_000, 4_500_000, 510_000, 4_510_000)
        crs = find_utm_crs(aoi, CRS.from_epsg(32611))
        # Centroid at ~-117° → zone 11
        assert crs == CRS.from_epsg(32611)

    def test_string_crs_input(self):
        """Should accept string CRS."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        crs = find_utm_crs(aoi, "EPSG:4326")
        assert crs == CRS.from_epsg(32612)


# ---------------------------------------------------------------------------
# make_opera_reference_grid
# ---------------------------------------------------------------------------


class TestMakeOperaReferenceGrid:
    """Tests for make_opera_reference_grid."""

    def test_basic_output_structure(self):
        """Grid has correct dims, CRS, and transform."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x")
        assert da.rio.crs is not None
        assert da.rio.crs.is_projected
        assert isinstance(da.rio.transform(), Affine)

    def test_resolution_is_30m(self):
        """All pixel spacings must be exactly OPERA_RESOLUTION (30 m)."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        dx = np.diff(da.x.values)
        dy = np.abs(np.diff(da.y.values))
        np.testing.assert_allclose(dx, OPERA_RESOLUTION, atol=0.01)
        np.testing.assert_allclose(dy, OPERA_RESOLUTION, atol=0.01)

    def test_bounds_snap_to_30m(self):
        """Grid bounds should be multiples of 30 m."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        bounds = da.rio.bounds()
        for b in bounds:
            assert b % OPERA_RESOLUTION == pytest.approx(0, abs=0.01), (
                f"Bound {b} is not snapped to {OPERA_RESOLUTION} m grid"
            )

    def test_grid_covers_aoi(self):
        """Grid extent must fully contain the reprojected AOI."""
        from pyproj import Transformer
        from shapely.ops import transform as shapely_transform

        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        # Reproject AOI to grid CRS for comparison
        t = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
        aoi_proj = shapely_transform(t.transform, aoi)

        grid_bounds = da.rio.bounds()  # (left, bottom, right, top)
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_proj.bounds

        assert grid_bounds[0] <= aoi_minx
        assert grid_bounds[1] <= aoi_miny
        assert grid_bounds[2] >= aoi_maxx
        assert grid_bounds[3] >= aoi_maxy

    def test_wgs84_aoi_reasonable_size(self):
        """A ~0.1° box should produce a grid of ~hundreds of pixels, not 1-2."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        # ~0.1° is roughly 8-11 km → ~270-370 pixels at 30 m
        assert da.shape[0] > 100, f"Height {da.shape[0]} too small for ~0.1° AOI"
        assert da.shape[1] > 100, f"Width {da.shape[1]} too small for ~0.1° AOI"

    def test_utm_aoi_input(self):
        """AOI already in UTM should work and produce correct grid."""
        aoi = box(500_000, 4_500_000, 510_000, 4_510_000)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:32612")

        # 10 km / 30 m ≈ 333-334 pixels
        assert 333 <= da.shape[0] <= 335
        assert 333 <= da.shape[1] <= 335
        assert da.rio.crs == CRS.from_epsg(32612)

    def test_y_decreasing_x_increasing(self):
        """Coordinates must be north-up: y decreasing, x increasing."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        assert np.all(np.diff(da.x.values) > 0)
        assert np.all(np.diff(da.y.values) < 0)

    def test_deterministic(self):
        """Same inputs produce identical grid."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da1 = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")
        da2 = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        np.testing.assert_array_equal(da1.x.values, da2.x.values)
        np.testing.assert_array_equal(da1.y.values, da2.y.values)
        assert da1.rio.transform() == da2.rio.transform()

    def test_fill_value_and_dtype(self):
        """Custom fill_value and dtype are respected."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        da = make_opera_reference_grid(
            aoi=aoi, aoi_crs="EPSG:4326",
            fill_value=-9999, dtype="int16",
        )
        assert da.dtype == np.int16
        assert np.all(da.values == -9999)

    def test_reproject_match_roundtrip(self):
        """A DataArray reprojected to the OPERA grid should match its shape."""
        aoi = box(-111.9, 40.7, -111.8, 40.8)
        ref = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        # Create a dummy source in same CRS
        src = xr.DataArray(
            np.ones((10, 10), dtype="float32"),
            dims=("y", "x"),
            coords={
                "y": np.linspace(ref.y.values[0], ref.y.values[9], 10),
                "x": np.linspace(ref.x.values[0], ref.x.values[9], 10),
            },
        )
        src = src.rio.write_crs(ref.rio.crs)
        src = src.rio.write_transform()

        out = src.rio.reproject_match(ref)
        assert out.shape == ref.shape
        assert out.rio.crs == ref.rio.crs


# ---------------------------------------------------------------------------
# Cross-UTM-zone handling
# ---------------------------------------------------------------------------


class TestCrossUtmZone:
    """Tests for AOIs that span multiple UTM zone boundaries."""

    def _check_grid_covers_aoi(self, aoi, aoi_crs, da):
        """Helper: verify grid fully covers the AOI."""
        t = Transformer.from_crs(aoi_crs, da.rio.crs, always_xy=True)
        aoi_proj = shapely_transform(t.transform, aoi)
        left, bottom, right, top = da.rio.bounds()
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_proj.bounds
        assert left <= aoi_minx, f"Grid left {left} > AOI left {aoi_minx}"
        assert bottom <= aoi_miny, f"Grid bottom {bottom} > AOI bottom {aoi_miny}"
        assert right >= aoi_maxx, f"Grid right {right} < AOI right {aoi_maxx}"
        assert top >= aoi_maxy, f"Grid top {top} < AOI top {aoi_maxy}"

    def test_aoi_spanning_two_utm_zones(self):
        """AOI crossing the UTM 5N/6N boundary (-150°) produces a valid grid."""
        # Straddles -150° longitude (zone 5/6 boundary)
        aoi = box(-150.3, 61.0, -149.7, 61.3)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        assert da.rio.crs.is_projected
        # Grid must still fully cover the AOI
        self._check_grid_covers_aoi(aoi, "EPSG:4326", da)
        # Resolution still 30 m
        np.testing.assert_allclose(np.diff(da.x.values), OPERA_RESOLUTION, atol=0.01)

    def test_wide_aoi_across_three_zones(self):
        """A wide AOI spanning ~18° of longitude (3 UTM zones)."""
        # Zones 10, 11, 12 in the western US
        aoi = box(-126.0, 45.0, -108.0, 46.0)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        self._check_grid_covers_aoi(aoi, "EPSG:4326", da)
        # Should be very wide: ~18° ≈ 1400 km → ~46000 pixels
        assert da.shape[1] > 30_000, f"Width {da.shape[1]} too small for 18° span"

    def test_cross_zone_utm_input(self):
        """AOI in UTM that extends into the neighboring zone."""
        # Far east edge of UTM 11N (past 114°W → zone 12)
        aoi = box(730_000, 4_800_000, 780_000, 4_850_000)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:32611")

        self._check_grid_covers_aoi(aoi, "EPSG:32611", da)
        assert da.rio.crs.is_projected

    def test_cross_zone_grid_resolution_uniform(self):
        """Pixel spacings remain exactly 30 m even for cross-zone grids."""
        aoi = box(-150.3, 61.0, -149.7, 61.3)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        dx = np.diff(da.x.values)
        dy = np.abs(np.diff(da.y.values))
        np.testing.assert_allclose(dx, OPERA_RESOLUTION, atol=0.01)
        np.testing.assert_allclose(dy, OPERA_RESOLUTION, atol=0.01)

    def test_cross_zone_bounds_snapped(self):
        """Bounds are still multiples of 30 m for cross-zone grids."""
        aoi = box(-150.3, 61.0, -149.7, 61.3)
        da = make_opera_reference_grid(aoi=aoi, aoi_crs="EPSG:4326")

        for b in da.rio.bounds():
            assert b % OPERA_RESOLUTION == pytest.approx(0, abs=0.01)


# ---------------------------------------------------------------------------
# Dataset structure validation against reference netCDF
# ---------------------------------------------------------------------------

REFERENCE_NC = Path(
    "/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/netcdfs/"
    "Turnagain_Pass_and_Girdwood/season_2024-2025_Turnagain_Pass_and_Girdwood.nc"
)

# Variables that assemble_dataset produces (before post-processing)
ASSEMBLE_VARS = {
    # (name, expected_spatial_dims)
    "VV":         ("time", "y", "x"),
    "VH":         ("time", "y", "x"),
    "lia_mask":   ("time", "y", "x"),
    "anf":        ("static_track", "y", "x"),
    "lia":        ("static_track", "y", "x"),
    "dem":        ("y", "x"),
    "slope":      ("y", "x"),
    "aspect":     ("y", "x"),
    "fcf":        ("y", "x"),
    "water_mask": ("y", "x"),
    "urban_mask": ("y", "x"),
}


@pytest.mark.skipif(not REFERENCE_NC.exists(), reason="Reference netCDF not available")
class TestDatasetStructureAgainstReference:
    """Verify that the reference dataset matches the expected assemble_dataset schema."""

    @pytest.fixture(autouse=True)
    def _load_reference(self):
        self.ds = xr.open_dataset(REFERENCE_NC)

    def test_all_assemble_vars_present(self):
        """Reference dataset contains every variable that assemble_dataset should produce."""
        missing = [v for v in ASSEMBLE_VARS if v not in self.ds.data_vars]
        assert not missing, f"Missing variables in reference: {missing}"

    @pytest.mark.parametrize("var,expected_dims", list(ASSEMBLE_VARS.items()))
    def test_variable_dims(self, var, expected_dims):
        """Each variable has the correct dimension names."""
        if var not in self.ds.data_vars:
            pytest.skip(f"{var} not in reference dataset")
        assert self.ds[var].dims == expected_dims, (
            f"{var}: expected dims {expected_dims}, got {self.ds[var].dims}"
        )

    def test_required_coords_present(self):
        """Reference dataset has the expected coordinates."""
        required = {"x", "y", "time", "static_track", "track", "direction", "platform"}
        present = set(self.ds.coords)
        missing = required - present
        assert not missing, f"Missing coords: {missing}"

    def test_x_monotonic_increasing(self):
        assert np.all(np.diff(self.ds.x.values) > 0)

    def test_y_monotonic(self):
        """y should be monotonically increasing or decreasing."""
        dy = np.diff(self.ds.y.values)
        assert np.all(dy > 0) or np.all(dy < 0)

    def test_time_sorted(self):
        times = self.ds.time.values
        assert np.all(times[:-1] <= times[1:])

    def test_static_track_has_values(self):
        assert len(self.ds.static_track) > 0

    def test_sar_vars_are_float32(self):
        for var in ("VV", "VH"):
            assert self.ds[var].dtype == np.float32, f"{var} dtype: {self.ds[var].dtype}"

    def test_no_all_nan_spatial_slices(self):
        """VV should have at least some valid data in each time step."""
        vv = self.ds["VV"]
        for t in range(min(3, len(vv.time))):
            assert not np.all(np.isnan(vv.isel(time=t).values)), (
                f"VV time={t} is entirely NaN"
            )
