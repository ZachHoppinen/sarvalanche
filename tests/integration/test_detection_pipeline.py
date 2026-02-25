"""
Integration test for the full run_detection pipeline.

Based on scripts/fast_example.py — uses the Jackson Hole / Teton area event
from 2020-01-11.  Requires the local data cache at
local/data/2020-01-11.nc and local/data/2020-01-11.gpkg to already exist
(populated by running fast_example.py once); the test passes overwrite=False
so no network calls or heavy computation are needed on CI.

Mark deliberately to allow skipping in environments without the cache:
    pytest -m "not slow" -v          # skips this file
    pytest tests/integration/ -v     # runs this file
"""
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Constants matching fast_example.py
# ---------------------------------------------------------------------------

AOI = box(-110.772, 43.734, -110.745, 43.756)
AVALANCHE_DATE = "2020-01-11"
LOCAL_CACHE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/data")

EXPECTED_VARS = [
    "detections",
    "p_pixelwise",
    "p_empirical",
    "p_fcf",
    "p_runout",
    "p_slope",
    "release_zones",
    "distance_mahalanobis",
]

PROBABILITY_VARS = [
    "p_pixelwise",
    "p_empirical",
    "p_fcf",
    "p_runout",
    "p_slope",
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detection_result():
    """Run (or load from cache) the full detection pipeline once per module."""
    from sarvalanche.detection_pipeline import run_detection

    cache_exists = (
        LOCAL_CACHE.exists()
        and (LOCAL_CACHE / "2020-01-11.nc").exists()
        and (LOCAL_CACHE / "2020-01-11.nc").stat().st_size > 0
        and (LOCAL_CACHE / "2020-01-11.gpkg").exists()
    )

    if cache_exists:
        cache_dir = LOCAL_CACHE
        overwrite = False
    else:
        pytest.skip(
            "Local data cache not found at local/data/2020-01-11.nc — "
            "run scripts/fast_example.py once to populate it, then re-run this test."
        )

    ds = run_detection(
        aoi=AOI,
        avalanche_date=AVALANCHE_DATE,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )
    return ds, cache_dir


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_returns_xarray_dataset(detection_result):
    ds, _ = detection_result
    assert isinstance(ds, xr.Dataset)


@pytest.mark.slow
def test_expected_variables_present(detection_result):
    ds, _ = detection_result
    missing = [v for v in EXPECTED_VARS if v not in ds.data_vars]
    assert missing == [], f"Missing variables: {missing}"


@pytest.mark.slow
def test_dataset_has_spatial_dims(detection_result):
    ds, _ = detection_result
    assert "y" in ds.dims
    assert "x" in ds.dims


@pytest.mark.slow
def test_dataset_has_time_dim(detection_result):
    ds, _ = detection_result
    assert "time" in ds.dims


@pytest.mark.slow
def test_dataset_has_crs(detection_result):
    ds, _ = detection_result
    # At least one spatial variable must carry a CRS
    for var in PROBABILITY_VARS:
        if var in ds.data_vars:
            assert ds[var].rio.crs is not None, f"{var} is missing a CRS"
            break


# ---------------------------------------------------------------------------
# Probability variables: range and finiteness
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("var", PROBABILITY_VARS)
def test_probability_var_in_unit_interval(detection_result, var):
    ds, _ = detection_result
    if var not in ds.data_vars:
        pytest.skip(f"{var} not in dataset")
    values = ds[var].values
    finite = values[np.isfinite(values)]
    assert finite.min() >= 0.0, f"{var} has values below 0"
    assert finite.max() <= 1.0, f"{var} has values above 1"


@pytest.mark.slow
@pytest.mark.parametrize("var", PROBABILITY_VARS)
def test_probability_var_has_finite_values(detection_result, var):
    ds, _ = detection_result
    if var not in ds.data_vars:
        pytest.skip(f"{var} not in dataset")
    assert np.isfinite(ds[var].values).any(), f"{var} is entirely NaN/Inf"


# ---------------------------------------------------------------------------
# Detections variable
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_detections_is_binary(detection_result):
    ds, _ = detection_result
    values = ds["detections"].values
    finite = values[np.isfinite(values)]
    unique_vals = np.unique(finite)
    assert set(unique_vals).issubset({0.0, 1.0}), (
        f"detections contains non-binary values: {unique_vals}"
    )


@pytest.mark.slow
def test_detections_has_some_positive(detection_result):
    """The known avalanche event should produce at least one detection."""
    ds, _ = detection_result
    assert (ds["detections"].values == 1).any(), "No avalanche detections found"


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_netcdf_output_exists(detection_result):
    _, cache_dir = detection_result
    nc_path = cache_dir / "2020-01-11.nc"
    assert nc_path.exists(), f"NetCDF not found at {nc_path}"
    assert nc_path.stat().st_size > 0, "NetCDF is empty"


@pytest.mark.slow
def test_geopackage_output_exists(detection_result):
    _, cache_dir = detection_result
    gpkg_path = cache_dir / "2020-01-11.gpkg"
    assert gpkg_path.exists(), f"GeoPackage not found at {gpkg_path}"
    assert gpkg_path.stat().st_size > 0, "GeoPackage is empty"


@pytest.mark.slow
@pytest.mark.parametrize("var", EXPECTED_VARS)
def test_geotiff_output_exists(detection_result, var):
    _, cache_dir = detection_result
    tif_path = cache_dir / "probabilities" / f"2020-01-11_{var}.tif"
    assert tif_path.exists(), f"GeoTIFF not found: {tif_path}"
    assert tif_path.stat().st_size > 0, f"GeoTIFF is empty: {tif_path}"


# ---------------------------------------------------------------------------
# Spatial sanity: dataset covers the requested AOI
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dataset_covers_aoi(detection_result):
    ds, _ = detection_result
    # AOI lon range: -110.772 to -110.745 ; lat range: 43.734 to 43.756
    # Check that x/y coordinates overlap with the AOI bounding box
    x_vals = ds.x.values
    y_vals = ds.y.values
    assert x_vals.min() <= -110.745
    assert x_vals.max() >= -110.772
    assert y_vals.min() <= 43.756
    assert y_vals.max() >= 43.734
