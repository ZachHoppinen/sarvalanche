"""Tests for CRS preservation across netCDF export/load round-trips."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from sarvalanche.io.export import export_netcdf
from sarvalanche.io.dataset import load_netcdf_to_dataset


def _make_dataset(crs="EPSG:32612"):
    """Small dataset that mimics assemble_dataset output."""
    ny, nx, nt = 10, 12, 3
    y = np.arange(4_500_000, 4_500_000 + ny * 30, 30, dtype="float64")[::-1]
    x = np.arange(500_000, 500_000 + nx * 30, 30, dtype="float64")

    ds = xr.Dataset(
        {
            "VV": (("time", "y", "x"), np.random.rand(nt, ny, nx).astype("float32")),
            "dem": (("y", "x"), np.random.rand(ny, nx).astype("float32")),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-13", "2024-01-25"], dtype="datetime64[ns]"),
            "y": y,
            "x": x,
        },
    )
    ds = ds.rio.write_crs(crs)
    return ds


class TestCrsRoundTrip:
    """CRS must survive export_netcdf → load_netcdf_to_dataset."""

    def test_dataset_crs_preserved(self, tmp_path):
        ds = _make_dataset("EPSG:32612")
        fp = tmp_path / "test.nc"
        export_netcdf(ds, fp)

        loaded = load_netcdf_to_dataset(fp)
        assert loaded.rio.crs is not None
        assert loaded.rio.crs == CRS.from_epsg(32612)

    def test_variable_crs_preserved(self, tmp_path):
        """Individual DataArrays must also carry the CRS after reload."""
        ds = _make_dataset("EPSG:32612")
        fp = tmp_path / "test.nc"
        export_netcdf(ds, fp)

        loaded = load_netcdf_to_dataset(fp)
        for var in ("VV", "dem"):
            assert loaded[var].rio.crs is not None, f"{var} lost its CRS"
            assert loaded[var].rio.crs == CRS.from_epsg(32612)

    def test_crs_attr_matches(self, tmp_path):
        ds = _make_dataset("EPSG:32606")
        fp = tmp_path / "test.nc"
        export_netcdf(ds, fp)

        loaded = load_netcdf_to_dataset(fp)
        assert loaded.attrs.get("crs") == "EPSG:32606"
        assert loaded.rio.crs == CRS.from_epsg(32606)

    def test_spatial_ref_is_coordinate(self, tmp_path):
        """spatial_ref should be a coordinate, not a data variable."""
        ds = _make_dataset("EPSG:32612")
        fp = tmp_path / "test.nc"
        export_netcdf(ds, fp)

        loaded = load_netcdf_to_dataset(fp)
        assert "spatial_ref" in loaded.coords

    def test_geographic_crs_preserved(self, tmp_path):
        """EPSG:4326 round-trips correctly (for older datasets)."""
        ds = _make_dataset("EPSG:4326")
        fp = tmp_path / "test.nc"
        export_netcdf(ds, fp)

        loaded = load_netcdf_to_dataset(fp)
        assert loaded.rio.crs == CRS.from_epsg(4326)
        assert loaded["VV"].rio.crs == CRS.from_epsg(4326)

    def test_crs_from_attr_fallback(self, tmp_path):
        """Older files without spatial_ref but with crs attr still load CRS."""
        ds = _make_dataset("EPSG:32612")
        fp = tmp_path / "test.nc"

        # Simulate an old-style export: just set attr, drop spatial_ref
        ds.attrs["crs"] = "EPSG:32612"
        if "spatial_ref" in ds.coords:
            ds = ds.drop_vars("spatial_ref")
        for var in ds.data_vars:
            ds[var].attrs.pop("grid_mapping", None)
        ds.to_netcdf(fp)

        loaded = load_netcdf_to_dataset(fp)
        assert loaded.rio.crs is not None
        assert loaded.rio.crs == CRS.from_epsg(32612)

    def test_double_roundtrip(self, tmp_path):
        """CRS survives two save/load cycles."""
        ds = _make_dataset("EPSG:32612")

        fp1 = tmp_path / "trip1.nc"
        export_netcdf(ds, fp1)
        ds1 = load_netcdf_to_dataset(fp1)

        fp2 = tmp_path / "trip2.nc"
        export_netcdf(ds1, fp2)
        ds2 = load_netcdf_to_dataset(fp2)

        assert ds2.rio.crs == CRS.from_epsg(32612)
        assert ds2["VV"].rio.crs == CRS.from_epsg(32612)
