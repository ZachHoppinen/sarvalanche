import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def projected_dem():
    """20×20 synthetic DEM with UTM CRS and 10 m pixels."""
    rows, cols = 20, 20
    # Flat base elevation
    data = np.full((rows, cols), 500.0)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": np.linspace(6_700_000, 6_700_190, rows),
            "x": np.linspace(500_000, 500_190, cols),
        },
    )
    da = da.rio.write_crs("EPSG:32633")
    da = da.rio.write_transform()
    return da


@pytest.fixture
def projected_dem_with_peak(projected_dem):
    """20×20 UTM DEM with a single elevated peak at the centre."""
    data = projected_dem.values.copy()
    data[10, 10] = 600.0  # 100 m above surroundings
    da = xr.DataArray(data, dims=projected_dem.dims, coords=projected_dem.coords)
    da = da.rio.write_crs(projected_dem.rio.crs)
    da = da.rio.write_transform()
    return da


@pytest.fixture
def projected_dem_with_hollow(projected_dem):
    """20×20 UTM DEM with a single deep hollow at the centre."""
    data = projected_dem.values.copy()
    data[10, 10] = 400.0  # 100 m below surroundings
    da = xr.DataArray(data, dims=projected_dem.dims, coords=projected_dem.coords)
    da = da.rio.write_crs(projected_dem.rio.crs)
    da = da.rio.write_transform()
    return da


@pytest.fixture
def projected_dem_with_nan(projected_dem):
    """20×20 UTM DEM with NaN in top-left corner."""
    data = projected_dem.values.copy()
    data[0, 0] = np.nan
    da = xr.DataArray(data, dims=projected_dem.dims, coords=projected_dem.coords)
    da = da.rio.write_crs(projected_dem.rio.crs)
    da = da.rio.write_transform()
    return da


@pytest.fixture
def geographic_dem():
    """10×10 DEM with geographic (WGS84) CRS — should raise in all functions."""
    data = np.ones((10, 10)) * 500.0
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": np.linspace(60.0, 60.09, 10),
            "x": np.linspace(25.0, 25.09, 10),
        },
    )
    da = da.rio.write_crs("EPSG:4326")
    return da


# ---------------------------------------------------------------------------
# compute_tpi
# ---------------------------------------------------------------------------

class TestComputeTpi:
    def test_output_shape_preserved(self, projected_dem):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem)
        assert tpi.shape == projected_dem.shape

    def test_flat_dem_is_zero(self, projected_dem):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem)
        # All values should be effectively 0 for a flat DEM
        assert np.allclose(np.nan_to_num(tpi.values), 0.0, atol=1e-6)

    def test_peak_is_positive(self, projected_dem_with_peak):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem_with_peak, radius_m=50.0)
        assert tpi.values[10, 10] > 0

    def test_hollow_is_negative(self, projected_dem_with_hollow):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem_with_hollow, radius_m=50.0)
        assert tpi.values[10, 10] < 0

    def test_nan_preserved(self, projected_dem_with_nan):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem_with_nan)
        assert np.isnan(tpi.values[0, 0])

    def test_geographic_raises(self, geographic_dem):
        from sarvalanche.utils.terrain import compute_tpi
        with pytest.raises(ValueError, match="projected"):
            compute_tpi(geographic_dem)

    def test_name(self, projected_dem):
        from sarvalanche.utils.terrain import compute_tpi
        tpi = compute_tpi(projected_dem)
        assert tpi.name == "tpi"


# ---------------------------------------------------------------------------
# compute_curvature
# ---------------------------------------------------------------------------

class TestComputeCurvature:
    def test_output_shape_preserved(self, projected_dem):
        from sarvalanche.utils.terrain import compute_curvature
        curv = compute_curvature(projected_dem)
        assert curv.shape == projected_dem.shape

    def test_flat_dem_is_zero(self, projected_dem):
        from sarvalanche.utils.terrain import compute_curvature
        curv = compute_curvature(projected_dem)
        interior = curv.values[1:-1, 1:-1]
        assert np.allclose(np.nan_to_num(interior), 0.0, atol=1e-3)

    def test_convex_hill_positive_at_peak(self, projected_dem_with_peak):
        from sarvalanche.utils.terrain import compute_curvature
        curv = compute_curvature(projected_dem_with_peak)
        assert curv.values[10, 10] > 0

    def test_geographic_raises(self, geographic_dem):
        from sarvalanche.utils.terrain import compute_curvature
        with pytest.raises(ValueError, match="projected"):
            compute_curvature(geographic_dem)

    def test_attrs_not_mutated(self, projected_dem):
        from sarvalanche.utils.terrain import compute_curvature
        original_attrs = dict(projected_dem.attrs)
        compute_curvature(projected_dem)
        assert projected_dem.attrs == original_attrs


# ---------------------------------------------------------------------------
# compute_flow_accumulation
# ---------------------------------------------------------------------------

class TestComputeFlowAccumulation:
    def test_output_shape_preserved(self, projected_dem):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        acc = compute_flow_accumulation(projected_dem)
        assert acc.shape == projected_dem.shape

    def test_all_finite_and_nonnegative(self, projected_dem):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        acc = compute_flow_accumulation(projected_dem)
        vals = acc.values
        finite_vals = vals[np.isfinite(vals)]
        assert len(finite_vals) > 0
        assert (finite_vals >= 0).all()

    def test_max_le_n_cells(self, projected_dem):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        acc = compute_flow_accumulation(projected_dem)
        n_cells = projected_dem.size
        assert float(np.nanmax(acc.values)) <= n_cells

    def test_nan_preserved(self, projected_dem_with_nan):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        acc = compute_flow_accumulation(projected_dem_with_nan)
        assert np.isnan(acc.values[0, 0])

    def test_geographic_raises(self, geographic_dem):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        with pytest.raises(ValueError, match="projected"):
            compute_flow_accumulation(geographic_dem)

    def test_name(self, projected_dem):
        from sarvalanche.utils.terrain import compute_flow_accumulation
        acc = compute_flow_accumulation(projected_dem)
        assert acc.name == "flow_accumulation"


# ---------------------------------------------------------------------------
# multiscale_ridgeline_tpi
# ---------------------------------------------------------------------------

class TestMultiscaleRidgelineTpi:
    def test_returns_expected_keys(self, projected_dem):
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(projected_dem, fine_radius_m=30.0, coarse_radius_m=80.0)
        assert set(result) == {"tpi_fine", "tpi_coarse", "ridgeline_mask"}

    def test_output_shapes_match_dem(self, projected_dem):
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(projected_dem, fine_radius_m=30.0, coarse_radius_m=80.0)
        for key in ("tpi_fine", "tpi_coarse", "ridgeline_mask"):
            assert result[key].shape == projected_dem.shape, f"{key} shape mismatch"

    def test_ridgeline_mask_is_binary(self, projected_dem):
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(projected_dem, fine_radius_m=30.0, coarse_radius_m=80.0)
        vals = result["ridgeline_mask"].values
        assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_flat_dem_has_no_ridgelines(self, projected_dem):
        """Perfectly flat DEM → TPI everywhere 0 → no pixels pass threshold."""
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(
            projected_dem,
            fine_radius_m=30.0,
            coarse_radius_m=80.0,
            ridge_threshold=2.0,
        )
        assert not result["ridgeline_mask"].values.any()

    def test_peak_is_detected_as_ridge(self, projected_dem_with_peak):
        """A prominent peak should be flagged as a ridgeline pixel."""
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(
            projected_dem_with_peak,
            fine_radius_m=30.0,
            coarse_radius_m=80.0,
            ridge_threshold=2.0,
            coarse_min=-5.0,
        )
        assert result["ridgeline_mask"].values[10, 10], "Peak pixel should be a ridgeline"

    def test_hollow_is_not_ridgeline(self, projected_dem_with_hollow):
        """A hollow (negative TPI) should never be classified as a ridgeline."""
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(
            projected_dem_with_hollow,
            fine_radius_m=30.0,
            coarse_radius_m=80.0,
            ridge_threshold=2.0,
        )
        assert not result["ridgeline_mask"].values[10, 10], "Hollow pixel should not be a ridgeline"

    def test_coarse_min_suppresses_valley_bumps(self, projected_dem):
        """Raising coarse_min to a high value should suppress all ridgelines."""
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        # Set ridge_threshold=0 so everything is a candidate, but coarse_min so high
        # that no pixel can satisfy it on a flat DEM (all coarse TPI ≈ 0).
        result = multiscale_ridgeline_tpi(
            projected_dem,
            fine_radius_m=30.0,
            coarse_radius_m=80.0,
            ridge_threshold=0.0,
            coarse_min=1000.0,
        )
        assert not result["ridgeline_mask"].values.any()

    def test_tpi_fine_name(self, projected_dem):
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        result = multiscale_ridgeline_tpi(projected_dem, fine_radius_m=30.0, coarse_radius_m=80.0)
        assert result["tpi_fine"].name == "tpi_fine"
        assert result["tpi_coarse"].name == "tpi_coarse"

    def test_geographic_raises(self, geographic_dem):
        from sarvalanche.utils.terrain import multiscale_ridgeline_tpi
        with pytest.raises(ValueError, match="projected"):
            multiscale_ridgeline_tpi(geographic_dem)
