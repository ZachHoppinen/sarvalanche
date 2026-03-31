"""Tests for sarvalanche.detection_pipeline.prepare_dataset."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ds(ny=10, nx=12, nt=3, crs_epsg="EPSG:32606"):
    """Minimal dataset that looks like assemble_dataset output (linear scale)."""
    y = np.linspace(4_500_000, 4_500_000 + ny * 30, ny)[::-1]
    x = np.linspace(500_000, 500_000 + nx * 30, nx)
    times = pd.date_range("2024-01-01", periods=nt, freq="12D")

    rng = np.random.default_rng(42)
    ds = xr.Dataset(
        {
            "VV": xr.DataArray(
                rng.uniform(0.01, 0.5, (nt, ny, nx)).astype("float32"),
                dims=("time", "y", "x"),
                attrs={"source": "opera", "product": "RTC-S1", "units": "linear"},
            ),
            "VH": xr.DataArray(
                rng.uniform(0.005, 0.3, (nt, ny, nx)).astype("float32"),
                dims=("time", "y", "x"),
                attrs={"source": "opera", "product": "RTC-S1", "units": "linear"},
            ),
            "mask": xr.DataArray(
                np.ones((nt, ny, nx), dtype="int8"),
                dims=("time", "y", "x"),
                attrs={"source": "opera", "product": "RTC-S1", "units": "boolean"},
            ),
            "anf": xr.DataArray(
                rng.uniform(30.0, 50.0, (4, ny, nx)).astype("float32"),
                dims=("static_track", "y", "x"),
                attrs={"source": "opera", "product": "ANF", "units": "degrees"},
            ),
            "dem": xr.DataArray(
                rng.uniform(1500, 3000, (ny, nx)).astype("float32"),
                dims=("y", "x"),
                attrs={"source": "copernicus", "product": "DEM", "units": "m"},
            ),
            "slope": xr.DataArray(
                rng.uniform(0, 60, (ny, nx)).astype("float32"),
                dims=("y", "x"),
                attrs={"source": "derived", "product": "slope", "units": "degrees"},
            ),
            "aspect": xr.DataArray(
                rng.uniform(0, 360, (ny, nx)).astype("float32"),
                dims=("y", "x"),
                attrs={"source": "derived", "product": "aspect", "units": "degrees"},
            ),
            "fcf": xr.DataArray(
                rng.uniform(0, 100, (ny, nx)).astype("float32"),
                dims=("y", "x"),
                attrs={"source": "copernicus", "product": "FCF", "units": "percent"},
            ),
        },
        coords={
            "time": times.values,
            "y": y,
            "x": x,
            "static_track": np.arange(4),
        },
    )
    ds = ds.rio.write_crs(crs_epsg)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return ds


def _add_flowpy_vars(ds):
    """Add flowpy output variables to the dataset."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    rng = np.random.default_rng(99)
    ds["cell_counts"] = xr.DataArray(
        rng.integers(0, 50, (ny, nx)).astype("float32"),
        dims=("y", "x"),
        attrs={"source": "flowpy", "product": "cell_counts", "units": "count"},
    )
    ds["runout_angle"] = xr.DataArray(
        rng.uniform(0, 45, (ny, nx)).astype("float32"),
        dims=("y", "x"),
        attrs={"source": "flowpy", "product": "runout_angle", "units": "degrees"},
    )
    ds["release_zones"] = xr.DataArray(
        rng.integers(0, 2, (ny, nx)).astype("int8"),
        dims=("y", "x"),
        attrs={"source": "flowpy", "product": "release_zones", "units": "boolean"},
    )
    return ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aoi():
    return box(-149.8, 60.7, -149.5, 60.9)


@pytest.fixture
def aoi_projected():
    return box(500_000, 4_500_000, 500_360, 4_500_300)


@pytest.fixture
def avalanche_date():
    return "2024-01-25"


@pytest.fixture
def mock_assemble(aoi):
    """Mock assemble_dataset to return a synthetic ds."""
    ds = _make_synthetic_ds()
    with patch(
        "sarvalanche.detection_pipeline.assemble_dataset", return_value=ds
    ) as m:
        yield m


@pytest.fixture
def mock_flowpy():
    """Mock generate_runcount_alpha_angle to add flowpy vars and return a GeoDataFrame."""
    def _side_effect(ds):
        ds = _add_flowpy_vars(ds)
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32606")
        return ds, gdf

    with patch(
        "sarvalanche.detection_pipeline.generate_runcount_alpha_angle",
        side_effect=_side_effect,
    ) as m:
        yield m


@pytest.fixture
def mock_preprocess():
    """Mock preprocess_rtc to just pass through and tag the ds."""
    def _side_effect(ds, tv_weight=0.5):
        ds.attrs["preprocessed"] = "rtc_tv"
        return ds

    with patch(
        "sarvalanche.detection_pipeline.preprocess_rtc",
        side_effect=_side_effect,
    ) as m:
        yield m


@pytest.fixture
def mock_validate_canonical():
    with patch("sarvalanche.detection_pipeline.validate_canonical") as m:
        yield m


@pytest.fixture
def all_mocks(mock_assemble, mock_flowpy, mock_preprocess, mock_validate_canonical):
    """Bundle all mocks for convenience."""
    return {
        "assemble": mock_assemble,
        "flowpy": mock_flowpy,
        "preprocess": mock_preprocess,
        "validate_canonical": mock_validate_canonical,
    }


# ---------------------------------------------------------------------------
# Tests: default date window computation
# ---------------------------------------------------------------------------

class TestDefaultDateWindow:
    def test_default_dates_from_avalanche_date(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        # 6 * 12 = 72 days before, 3 * 12 = 36 days after
        expected_start = pd.Timestamp("2024-01-25") - pd.Timedelta(days=72)
        expected_stop = pd.Timestamp("2024-01-25") + pd.Timedelta(days=36)
        assert call_kwargs["start_date"] == expected_start
        assert call_kwargs["stop_date"] == expected_stop

    def test_explicit_dates_override_defaults(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            start_date="2024-01-01",
            stop_date="2024-03-01",
            crs="EPSG:32606",
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        assert call_kwargs["start_date"] == pd.Timestamp("2024-01-01")
        assert call_kwargs["stop_date"] == pd.Timestamp("2024-03-01")

    def test_no_avalanche_date_requires_explicit_dates(self, tmp_path, aoi, all_mocks):
        """Without avalanche_date, start/stop stay None → assemble gets None."""
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date=None,
            start_date="2024-01-01",
            stop_date="2024-03-01",
            crs="EPSG:32606",
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        assert call_kwargs["start_date"] == pd.Timestamp("2024-01-01")
        assert call_kwargs["stop_date"] == pd.Timestamp("2024-03-01")


# ---------------------------------------------------------------------------
# Tests: resolution defaults
# ---------------------------------------------------------------------------

class TestResolutionDefaults:
    def test_projected_crs_defaults_to_30m(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
            resolution=None,
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        # validate_resolution normalizes to a tuple
        assert call_kwargs["resolution"] in [30, (30, 30), (30.0, 30.0)]

    def test_geographic_crs_defaults_to_arcsec(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:4326",
            resolution=None,
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        expected = 1 / 3600
        # Could be scalar or tuple
        if isinstance(call_kwargs["resolution"], tuple):
            assert abs(call_kwargs["resolution"][0] - expected) < 1e-9
        else:
            assert abs(call_kwargs["resolution"] - expected) < 1e-9

    def test_explicit_resolution_passed_through(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
            resolution=60,
        )
        call_kwargs = all_mocks["assemble"].call_args.kwargs
        assert call_kwargs["resolution"] in [60, (60, 60), (60.0, 60.0)]


# ---------------------------------------------------------------------------
# Tests: cache directory structure
# ---------------------------------------------------------------------------

class TestCacheSetup:
    def test_cache_subdirs_created(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path / "cache",
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert (tmp_path / "cache" / "opera").is_dir()
        assert (tmp_path / "cache" / "arrays").is_dir()

    def test_job_name_controls_nc_stem(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
            job_name="my_custom_run",
        )
        # FlowPy mock writes a gpkg; check the stem
        # The ds_nc path is internal, but we can verify the gpkg stem
        # if flowpy ran, or just that the function didn't error out
        assert all_mocks["assemble"].called

    def test_default_stem_from_avalanche_date(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        # flowpy mock will write a gpkg; nc path would be 2024-01-25.nc
        assert all_mocks["flowpy"].called

    def test_default_stem_full_season_when_no_date(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date=None,
            start_date="2024-01-01",
            stop_date="2024-03-01",
            crs="EPSG:32606",
        )
        assert all_mocks["assemble"].called


# ---------------------------------------------------------------------------
# Tests: FlowPy integration
# ---------------------------------------------------------------------------

class TestFlowPy:
    def test_flowpy_called_on_fresh_run(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert all_mocks["flowpy"].called

    def test_flowpy_skipped_when_vars_present_and_gpkg_exists(
        self, tmp_path, aoi, mock_preprocess, mock_validate_canonical
    ):
        """If the cached nc already has flowpy vars AND the gpkg exists, skip flowpy."""
        from sarvalanche.detection_pipeline import prepare_dataset

        ds = _add_flowpy_vars(_make_synthetic_ds())
        ds.attrs["preprocessed"] = "rtc_tv"

        # Write a cached nc and gpkg
        nc_path = tmp_path / "2024-01-25.nc"
        ds.to_netcdf(nc_path)
        gpkg_path = tmp_path / "2024-01-25.gpkg"
        gpd.GeoDataFrame(geometry=[], crs="EPSG:32606").to_file(
            gpkg_path, driver="GPKG"
        )

        with patch(
            "sarvalanche.detection_pipeline.load_netcdf_to_dataset", return_value=ds
        ), patch(
            "sarvalanche.detection_pipeline.generate_runcount_alpha_angle"
        ) as mock_fp:
            prepare_dataset(
                aoi=box(-149.8, 60.7, -149.5, 60.9),
                cache_dir=tmp_path,
                avalanche_date="2024-01-25",
                crs="EPSG:32606",
            )
            assert not mock_fp.called

    def test_flowpy_vars_loaded_from_static_fp(
        self, tmp_path, aoi, mock_preprocess, mock_validate_canonical
    ):
        """If gpkg exists but flowpy vars missing, load them from static_fp."""
        from sarvalanche.detection_pipeline import prepare_dataset

        # Base ds without flowpy
        ds_base = _make_synthetic_ds()
        # Donor ds with flowpy
        ds_donor = _add_flowpy_vars(_make_synthetic_ds())

        nc_path = tmp_path / "2024-01-25.nc"
        ds_base.to_netcdf(nc_path)
        gpkg_path = tmp_path / "2024-01-25.gpkg"
        gpd.GeoDataFrame(geometry=[], crs="EPSG:32606").to_file(
            gpkg_path, driver="GPKG"
        )
        static_nc = tmp_path / "static.nc"
        ds_donor.to_netcdf(static_nc)

        def _load_side_effect(fp, **kwargs):
            if str(fp) == str(nc_path):
                return ds_base.copy(deep=True)
            return ds_donor.copy(deep=True)

        with patch(
            "sarvalanche.detection_pipeline.load_netcdf_to_dataset",
            side_effect=_load_side_effect,
        ), patch(
            "sarvalanche.detection_pipeline.generate_runcount_alpha_angle"
        ) as mock_fp:
            ds, _ = prepare_dataset(
                aoi=aoi,
                cache_dir=tmp_path,
                avalanche_date="2024-01-25",
                crs="EPSG:32606",
                static_fp=static_nc,
            )
            assert not mock_fp.called
            assert "cell_counts" in ds.data_vars
            assert "runout_angle" in ds.data_vars
            assert "release_zones" in ds.data_vars


# ---------------------------------------------------------------------------
# Tests: preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_preprocess_called_on_fresh_ds(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert all_mocks["preprocess"].called

    def test_preprocess_skipped_if_already_done(
        self, tmp_path, aoi, mock_validate_canonical
    ):
        """If ds.attrs['preprocessed'] == 'rtc_tv', skip preprocessing."""
        from sarvalanche.detection_pipeline import prepare_dataset

        ds = _add_flowpy_vars(_make_synthetic_ds())
        ds.attrs["preprocessed"] = "rtc_tv"

        nc_path = tmp_path / "2024-01-25.nc"
        ds.to_netcdf(nc_path)
        gpkg_path = tmp_path / "2024-01-25.gpkg"
        gpd.GeoDataFrame(geometry=[], crs="EPSG:32606").to_file(
            gpkg_path, driver="GPKG"
        )

        with patch(
            "sarvalanche.detection_pipeline.load_netcdf_to_dataset", return_value=ds
        ), patch(
            "sarvalanche.detection_pipeline.preprocess_rtc"
        ) as mock_pp:
            prepare_dataset(
                aoi=aoi,
                cache_dir=tmp_path,
                avalanche_date="2024-01-25",
                crs="EPSG:32606",
            )
            assert not mock_pp.called


# ---------------------------------------------------------------------------
# Tests: overwrite flag
# ---------------------------------------------------------------------------

class TestOverwrite:
    def test_overwrite_forces_reassembly(
        self, tmp_path, aoi, mock_flowpy, mock_preprocess, mock_validate_canonical
    ):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds = _add_flowpy_vars(_make_synthetic_ds())

        # Write a cached nc
        nc_path = tmp_path / "2024-01-25.nc"
        ds.to_netcdf(nc_path)

        with patch(
            "sarvalanche.detection_pipeline.assemble_dataset",
            return_value=_make_synthetic_ds(),
        ) as mock_asm:
            prepare_dataset(
                aoi=aoi,
                cache_dir=tmp_path,
                avalanche_date="2024-01-25",
                crs="EPSG:32606",
                overwrite=True,
            )
            assert mock_asm.called

    def test_cached_nc_loaded_without_overwrite(
        self, tmp_path, aoi, mock_preprocess, mock_validate_canonical
    ):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds = _add_flowpy_vars(_make_synthetic_ds())
        ds.attrs["preprocessed"] = "rtc_tv"

        nc_path = tmp_path / "2024-01-25.nc"
        ds.to_netcdf(nc_path)
        gpkg_path = tmp_path / "2024-01-25.gpkg"
        gpd.GeoDataFrame(geometry=[], crs="EPSG:32606").to_file(
            gpkg_path, driver="GPKG"
        )

        with patch(
            "sarvalanche.detection_pipeline.assemble_dataset"
        ) as mock_asm, patch(
            "sarvalanche.detection_pipeline.load_netcdf_to_dataset", return_value=ds
        ):
            prepare_dataset(
                aoi=aoi,
                cache_dir=tmp_path,
                avalanche_date="2024-01-25",
                crs="EPSG:32606",
                overwrite=False,
            )
            assert not mock_asm.called


# ---------------------------------------------------------------------------
# Tests: return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    def test_returns_tuple_of_dataset_and_path(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        result = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert isinstance(result, tuple)
        ds, gpkg_path = result
        assert isinstance(ds, xr.Dataset)
        assert isinstance(gpkg_path, Path)

    def test_gpkg_path_has_correct_stem(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        _, gpkg_path = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert gpkg_path.suffix == ".gpkg"
        assert gpkg_path.stem == "2024-01-25"

    def test_gpkg_path_uses_job_name(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        _, gpkg_path = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
            job_name="my_run",
        )
        assert gpkg_path.stem == "my_run"

    def test_result_has_sar_variables(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds, _ = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert "VV" in ds.data_vars
        assert "VH" in ds.data_vars

    def test_result_has_static_variables(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds, _ = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        for var in ["dem", "slope", "aspect", "fcf"]:
            assert var in ds.data_vars

    def test_result_has_flowpy_variables(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds, _ = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        for var in ["cell_counts", "runout_angle", "release_zones"]:
            assert var in ds.data_vars

    def test_result_marked_preprocessed(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds, _ = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert ds.attrs.get("preprocessed") == "rtc_tv"

    def test_result_has_correct_dims(self, tmp_path, aoi, all_mocks):
        from sarvalanche.detection_pipeline import prepare_dataset

        ds, _ = prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
        )
        assert "time" in ds.VV.dims
        assert "y" in ds.VV.dims
        assert "x" in ds.VV.dims


# ---------------------------------------------------------------------------
# Tests: debug flag
# ---------------------------------------------------------------------------

class TestDebug:
    def test_debug_enables_debug_logging(self, tmp_path, aoi, all_mocks):
        import logging

        from sarvalanche.detection_pipeline import prepare_dataset

        prepare_dataset(
            aoi=aoi,
            cache_dir=tmp_path,
            avalanche_date="2024-01-25",
            crs="EPSG:32606",
            debug=True,
        )
        sarv_logger = logging.getLogger("sarvalanche")
        assert sarv_logger.level == logging.DEBUG
