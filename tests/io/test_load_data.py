import pytest
from unittest.mock import MagicMock

import hashlib
import warnings
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from sarvalanche.io.loaders.BaseLoader import BaseLoader
from sarvalanche.utils.constants import CANONICAL_DIMS_2D, REQUIRED_ATTRS

class DummyLoader(BaseLoader):
    sensor = "S1"
    product = "RTC"
    units = "m"

    def _open_file(self, path):
        # Simple DataArray with y/x dims
        da = xr.DataArray(
            np.ones((2, 2)) * int(path.stem[-1]),  # vary by file to test stacking
            dims=("y", "x")
        )
        da = da.rio.write_crs('EPSG:4326')
        return da

    def _parse_time(self, path):
        # Return a unique time per file for stacking tests
        idx = int(path.stem[-1])  # assume path name ends with 0,1,2...
        return pd.Timestamp(f"2020-01-0{idx+1}T00:00:00")


@pytest.fixture
def tmp_files(tmp_path):
    f1 = tmp_path / "a_VV.tif"
    f2 = tmp_path / "a_VH.tif"

    f1.write_bytes(b"x" * 1000)
    f2.write_bytes(b"x" * 1000)

    return [f1, f2]

@pytest.fixture
def tmp_files_with_outlier(tmp_path):
    f1 = tmp_path / "a_VV.tif"
    f2 = tmp_path / "a_VH.tif"

    f1.write_bytes(b"x" * 1000)
    f2.write_bytes(b"x" * 50_000)

    return [f1, f2]

def test_validate_files_empty_list():
    loader = DummyLoader()

    with pytest.raises(ValueError, match="No files downloaded"):
        loader._validate_files([])

def test_validate_files_missing_file(tmp_path):
    loader = DummyLoader()
    missing = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        loader._validate_files([missing])

def test_validate_files_empty_file(tmp_path):
    loader = DummyLoader()
    f = tmp_path / "empty.tif"
    f.touch()

    with pytest.raises(ValueError, match="Empty file"):
        loader._validate_files([f])

def test_validate_files_warns_on_small_file(tmp_path):
    loader = DummyLoader()
    f = tmp_path / "small.tif"
    f.write_bytes(b"x" * 10)

    with pytest.warns(RuntimeWarning, match="unusually small"):
        loader._validate_files([f])

def test_validate_files_size_outlier_warning(tmp_files_with_outlier):
    loader = DummyLoader()

    with pytest.warns(RuntimeWarning, match="Size outlier detected"):
        loader._validate_files(tmp_files_with_outlier)

def test_validate_files_grouping_regex(tmp_path):
    loader = DummyLoader()

    f1 = tmp_path / "OPERA_T001_foo_VV.tif"
    f2 = tmp_path / "OPERA_T001_foo_VH.tif"

    f1.write_bytes(b"x" * 1000)
    f2.write_bytes(b"x" * 5000)

    with pytest.warns(RuntimeWarning):
        loader._validate_files(
            [f1, f2],
            group_regex=r"T\d{3}",
        )

def test_validate_files_checksum_mismatch(tmp_files):
    loader = DummyLoader()

    bad_checksum = {
        tmp_files[0].name: "deadbeef",
    }

    with pytest.raises(ValueError, match="Checksum mismatch"):
        loader._validate_files(
            tmp_files,
            checksums=bad_checksum,
        )

def test_validate_files_checksum_valid(tmp_files):
    loader = DummyLoader()

    checksums = {}
    for f in tmp_files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        checksums[f.name] = h

    loader._validate_files(tmp_files, checksums=checksums)

def test_download_raises_if_cache_dir_none():
    loader = DummyLoader(cache_dir=None)

    with pytest.raises(ValueError, match="cache_dir must be set"):
        loader._download(["https://example.com/file.tif"])

def test_download_calls_parallel_downloader(mocker, tmp_path):
    loader = DummyLoader(cache_dir=tmp_path)

    urls = [
        "https://example.com/a.tif",
        "https://example.com/b.tif",
    ]

    expected = [tmp_path / "a.tif", tmp_path / "b.tif"]

    mock_download = mocker.patch(
        "sarvalanche.io.loaders.BaseLoader.download_urls_parallel",
        return_value=expected,
    )

    out = loader._download(urls)

    mock_download.assert_called_once_with(
        urls,
        out_directory=tmp_path,
    )

    assert out == expected

def test_download_empty_url_list(mocker, tmp_path):
    loader = DummyLoader(cache_dir=tmp_path)

    mock_download = mocker.patch(
        "sarvalanche.io.loaders.BaseLoader.download_urls_parallel",
        return_value=[],
    )

    out = loader._download([])

    mock_download.assert_called_once_with([], out_directory=tmp_path)
    assert out == []

def test_download_propagates_exception(mocker, tmp_path):
    loader = DummyLoader(cache_dir=tmp_path)

    mocker.patch(
        "sarvalanche.io.loaders.BaseLoader.download_urls_parallel",
        side_effect=RuntimeError("network down"),
    )

    with pytest.raises(RuntimeError, match="network down"):
        loader._download(["https://example.com/a.tif"])

def test_normalize_dims_renames_lat_lon():
    loader = DummyLoader()

    da = xr.DataArray(
        np.zeros((2, 3)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": [10, 11],
            "longitude": [20, 21, 22],
        },
    )

    out = loader._normalize_dims(da)

    assert out.dims == ("y", "x")
    assert "latitude" not in out.dims
    assert "longitude" not in out.dims

def test_normalize_dims_noop_if_already_yx():
    loader = DummyLoader()

    da = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
    )

    out = loader._normalize_dims(da)

    assert out is da  # renamed returns new object
    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out.values, da.values)

def test_normalize_dims_missing_x_or_y_raises():
    loader = DummyLoader()

    da = xr.DataArray(
        np.zeros((2, 2)),
        dims=("latitude", "time"),
    )

    with pytest.raises(ValueError, match="Spatial dims missing"):
        loader._normalize_dims(da)

def test_normalize_dims_no_spatial_dims_raises():
    loader = DummyLoader()

    da = xr.DataArray(
        np.zeros((3,)),
        dims=("time",),
    )

    with pytest.raises(ValueError, match="Spatial dims missing"):
        loader._normalize_dims(da)

def test_normalize_dims_preserves_extra_dims():
    loader = DummyLoader()

    da = xr.DataArray(
        np.zeros((4, 2, 3)),
        dims=("time", "latitude", "longitude"),
    )

    out = loader._normalize_dims(da)

    assert out.dims == ("time", "y", "x")

def test_assign_time_noop_if_time_coord_exists(mocker):
    loader = DummyLoader()

    da = xr.DataArray(
        np.zeros((1, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": ["2020-01-01"]},
    )

    out = loader._assign_time(da, Path("dummy.tif"))

    assert out is da
    assert "time" in out.coords
    assert out.sizes["time"] == 1

def test_assign_time_adds_time_dim(mocker):
    loader = DummyLoader()

    mock_time = np.datetime64("2020-01-02")
    mocker.patch.object(loader, "_parse_time", return_value=mock_time)

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
    )

    out = loader._assign_time(da, Path("file.tif"))

    assert "time" in out.dims
    assert out.sizes["time"] == 1
    assert out.coords["time"].values[0] == mock_time

def test_assign_time_adds_time_dim(mocker):
    loader = DummyLoader()

    mock_time = np.datetime64("2020-01-02")
    mocker.patch.object(loader, "_parse_time", return_value=mock_time)

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
    )

    out = loader._assign_time(da, Path("file.tif"))

    assert "time" in out.dims
    assert out.sizes["time"] == 1
    assert out.coords["time"].values[0] == mock_time

def test_assign_time_calls_parse_time_with_path(mocker):
    loader = DummyLoader()

    spy = mocker.patch.object(
        loader,
        "_parse_time",
        return_value=np.datetime64("2022-01-01"),
    )

    path = Path("some_file.tif")
    da = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))

    loader._assign_time(da, path)

    spy.assert_called_once_with(path)

def test_stack_single_array_returns_same():
    loader = DummyLoader()

    da = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))

    out = loader._stack([da])

    assert out is da

def test_stack_multiple_with_time():
    loader = DummyLoader()

    da1 = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": ["2020-01-01"]},
    )
    da2 = xr.DataArray(
        np.ones((1, 2, 2)) * 2,
        dims=("time", "y", "x"),
        coords={"time": ["2020-01-02"]},
    )

    out = loader._stack([da1, da2])

    assert "time" in out.dims
    assert out.sizes["time"] == 2
    np.testing.assert_array_equal(out.isel(time=0).values, da1.squeeze("time").values)
    np.testing.assert_array_equal(out.isel(time=1).values, da2.squeeze("time").values)

def test_stack_multiple_no_time_raises():
    loader = DummyLoader()

    da1 = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))
    da2 = xr.DataArray(np.ones((2, 2)) * 2, dims=("y", "x"))

    with pytest.raises(ValueError, match="Multiple arrays but no time dimension"):
        loader._stack([da1, da2])

@pytest.fixture
def sample_da_4326():
    """Small DataArray with x/y coordinates and CRS for reproject tests"""
    data = np.ones((2, 2))
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0],
            "x": [10.0, 11.0],
        }
    )
    da = da.rio.write_crs("EPSG:4326")
    return da

@pytest.fixture
def sample_da_3857():
    data = np.ones((2, 2))
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]}
    )
    return da.rio.write_crs("EPSG:3857")


def test_reproject_match_reference(sample_da_4326):
    loader = DummyLoader()
    # create a fake reference grid
    ref_grid = sample_da_4326.copy()
    loader.reference_grid = ref_grid.rio.write_crs('EPSG:4326')

    out = loader._reproject(sample_da_4326)
    # rioxarray reproject_match returns a DataArray
    assert isinstance(out, xr.DataArray)

def test_reproject_target_crs(sample_da_4326):
    loader = DummyLoader()
    loader.target_crs = "EPSG:3857"

    out = loader._reproject(sample_da_4326)
    assert isinstance(out, xr.DataArray)
    assert out.rio.crs.to_string() == "EPSG:3857"

def test_reproject_no_change(sample_da_4326):
    loader = DummyLoader()
    # no reference grid or target CRS
    out = loader._reproject(sample_da_4326)
    # should return the original
    assert out.equals(sample_da_4326)

def test_reproject_skips_if_crs_matches(sample_da_4326):
    loader = DummyLoader()
    loader.target_crs = "EPSG:4326"

    out = loader._reproject(sample_da_4326)
    # Should return the original object if CRS matches
    assert out.rio.crs.to_string() == "EPSG:4326"
    # Data values should remain identical
    np.testing.assert_array_equal(out.values, sample_da_4326.values)
    # Shape should be identical
    assert out.shape == sample_da_4326.shape

def test_reference_grid_takes_precedence(sample_da_4326, sample_da_3857):
    loader = DummyLoader()
    loader.reference_grid = sample_da_3857
    loader.target_crs = "EPSG:32633"  # different than reference grid

    out = loader._reproject(sample_da_4326)
    # Should match reference grid, not target_crs
    assert out.rio.crs == "EPSG:32633"

@pytest.fixture
def two_files_da_4326():
    """Two different DataArrays to simulate multiple files with same CRS."""
    da1 = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [10.0, 11.0]}
    ).rio.write_crs("EPSG:4326")
    da2 = xr.DataArray(
        np.ones((2, 2)) * 2,
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [10.0, 11.0]}
    ).rio.write_crs("EPSG:4326")
    return da1, da2

def test_first_file_sets_reference(two_files_da_4326):
    da1, da2 = two_files_da_4326
    loader = DummyLoader()

    # reference grid initially None
    assert loader.reference_grid is None

    # _reproject should now set first DA as reference
    out1 = loader._reproject(da1)
    assert loader.reference_grid is da1
    np.testing.assert_array_equal(out1.values, da1.values)
    assert out1.dims == da1.dims

    # second file should reproject to first file automatically
    out2 = loader._reproject(da2)
    assert out2.rio.crs == da1.rio.crs
    assert out2.shape == da1.shape
    # Values may not be exactly equal if interpolation happened; just check type
    assert isinstance(out2, xr.DataArray)

def test_reproject_sets_reference(two_files_da_4326):
    da1, da2 = two_files_da_4326
    loader = DummyLoader()
    loader.reference_grid = None
    out = loader._reproject(da1)
    assert (loader.reference_grid == da1).all()

def test_reproject_skips_if_first_file_has_reference(two_files_da_4326):
    da1, da2 = two_files_da_4326
    loader = DummyLoader()
    loader.reference_grid = da1  # already set manually

    out = loader._reproject(da1)
    # Should skip reproject, return same object
    np.testing.assert_array_equal(out.values, da1.values)
    assert out.dims == da1.dims

def test_reproject_match_vs_target_crs(two_files_da_4326):
    da1, da2 = two_files_da_4326
    loader = DummyLoader()
    loader.reference_grid = da1
    loader.target_crs = "EPSG:3857"  # different CRS
    out = loader._reproject(da2)
    # Should match reference_grid CRS, not target_crs
    assert out.rio.crs == "EPSG:3857"


@pytest.fixture
def sample_da():
    da = xr.DataArray(
        [[1, 2], [3, 4]],
        dims=("y", "x"),
        attrs={"original": "attr"}
    )
    return da


def test_add_attrs_basic(sample_da):
    loader = DummyLoader()
    urls = ["https://example.com/A_VV.tif", "https://example.com/B_VH.tif"]

    out = loader._add_attrs(sample_da, urls)

    # Original attributes preserved
    assert "original" in out.attrs
    assert out.attrs["original"] == "attr"

    # Standard loader attributes
    assert out.attrs["sensor"] == "S1"
    assert out.attrs["product"] == "RTC"
    assert out.attrs["units"] == "m"

    # Source URLs
    assert out.attrs["source_urls"] == urls

def test_add_attrs_with_crs(tmp_path):
    loader = DummyLoader()
    urls = ["https://example.com/A_VV.tif"]

    # Create a minimal xarray DataArray with rioxarray CRS
    da = xr.DataArray([[1, 2], [3, 4]], dims=("y", "x"))
    da = da.rio.write_crs("EPSG:4326")  # this properly sets the read-only accessor

    out = loader._add_attrs(da, urls)

    assert out.attrs["crs"] == "EPSG:4326"
    assert out.attrs["sensor"] == "S1"
    assert out.attrs["product"] == "RTC"

@pytest.fixture
def basic_da():
    return xr.DataArray([[1, 2], [3, 4]], dims=("y", "x"), attrs={"original": "value"})


@pytest.fixture
def da_with_crs():
    da = xr.DataArray([[1, 2], [3, 4]], dims=("y", "x"), attrs={"original": "value"})
    da = da.rio.write_crs("EPSG:4326")
    return da


def test_add_attrs_with_units(basic_da):
    loader = DummyLoader()
    loader.units = "m"
    urls = ["url1"]
    out = loader._add_attrs(basic_da, urls)

    assert out.attrs["units"] == "m"
    assert out.attrs["source_urls"] == urls

def test_add_attrs_empty_urls(basic_da):
    loader = DummyLoader()
    urls = []
    out = loader._add_attrs(basic_da, urls)

    assert out.attrs["source_urls"] == []
    assert "crs" not in out.attrs
    assert out.attrs["sensor"] == "S1"
    assert out.attrs["product"] == "RTC"


def test_add_attrs_preserve_existing_attributes(da_with_crs):
    loader = DummyLoader()
    loader.units = "m"
    urls = ["url1", "url2"]

    da_with_crs.attrs["existing"] = 123
    out = loader._add_attrs(da_with_crs, urls)

    # Original + new attributes
    assert out.attrs["existing"] == 123
    assert out.attrs["original"] == "value"
    assert out.attrs["sensor"] == "S1"
    assert out.attrs["product"] == "RTC"
    assert out.attrs["units"] == "m"
    assert out.attrs["source_urls"] == urls
    assert out.attrs["crs"] == "EPSG:4326"

@pytest.fixture
def base_da():
    da = xr.DataArray(
        [[1, 2], [3, 4]],
        dims=("y", "x"),
        attrs={
            "sensor": "S1",
            "product": "RTC",
            "source_urls": ["url1"],
            "units": "m",
            "crs": "EPSG:4326"
        },
    )
    return da


def test_validate_output_pass(base_da):
    loader = DummyLoader()
    # Should not raise
    loader._validate_output(base_da)


def test_validate_output_missing_dims():
    loader = DummyLoader()
    da = xr.DataArray([1, 2], dims=("y",))  # missing 'x'
    with pytest.raises(ValueError, match="Output missing spatial dims"):
        loader._validate_output(da)


def test_validate_output_empty_time():
    loader = DummyLoader()
    da = xr.DataArray([[[]]], dims=("time", "y", "x"), attrs={
        "sensor": "S1",
        "product": "RTC",
        "source_urls": ["url1"],
        "units": "m",
        "crs": "EPSG:4326"
    })
    with pytest.raises(ValueError, match="DataArray is empty"):
        loader._validate_output(da)


def test_validate_output_missing_attrs():
    loader = DummyLoader()
    da = xr.DataArray([[1, 2], [3, 4]], dims=("y", "x"), attrs={"sensor": "S1"})
    with pytest.raises(ValueError, match="Missing attrs"):
        loader._validate_output(da)


def test_validate_output_time_ok():
    loader = DummyLoader()
    da = xr.DataArray(
        [[[1, 2], [3, 4]]],  # shape (1,2,2)
        dims=("time", "y", "x"),
        attrs={
            "sensor": "S1",
            "product": "RTC",
            "source_urls": ["url1"],
            "units": "m",
            "crs": "EPSG:4326",
        },
    )
    # Should not raise
    loader._validate_output(da)

def test_validate_output_all_nans():
    loader = DummyLoader(cache_dir="/tmp")
    da = xr.DataArray(
        np.full((2, 2), np.nan),
        dims=("y", "x"),
        attrs={
            "sensor": "S1",
            "product": "RTC",
            "source_urls": ["url1"],
            "units": "m",
            "crs": "EPSG:4326",
        }
    )
    with pytest.raises(ValueError, match="DataArray contains only NaNs"):
        loader._validate_output(da)

@pytest.fixture
def loader(tmp_path):
    return DummyLoader(cache_dir=tmp_path)

def test_load_method(loader, tmp_path):
    urls = [f"url{i}" for i in range(2)]
    files = [tmp_path / f"file{i}.tif" for i in range(2)]
    for f in files:
        f.write_bytes(b"fake")  # non-empty file

    # Patch _download to return our fake files
    loader._download = MagicMock(return_value=files)

    # Now call load
    out = loader.load(urls)

    # Check that _download was called with the urls
    loader._download.assert_called_once_with(urls)

    # Check final DataArray has 'time', 'y', 'x'
    assert set(out.dims) == {"time", "y", "x"}
    assert out.sizes["time"] == len(files)

    # Check values match the dummy file values
    for i, t in enumerate(out["time"].values):
        np.testing.assert_array_equal(out.sel(time=t).values, np.ones((2, 2)) * i)

    # Check attributes
    for attr in ["sensor", "product", "units", "source_urls"]:
        assert attr in out.attrs

    assert out.attrs["source_urls"] == urls

@pytest.fixture
def da_yx():
    return xr.DataArray(np.ones((2, 2)), dims=("y", "x"))

@pytest.fixture
def da_time_yx():
    return xr.DataArray(np.ones((1, 2, 2)), dims=("time", "y", "x"), coords={"time": [pd.Timestamp("2020-01-01")]})

def test_assign_time_existing_time(loader, da_time_yx):
    # Should not change if time already exists
    out = loader._assign_time(da_time_yx, Path("dummy.tif"))
    assert "time" in out.dims
    assert out.equals(da_time_yx)  # unchanged

def test_assign_time_adds_time(loader, da_yx):
    # _parse_time returns a valid timestamp
    out = loader._assign_time(da_yx, Path("dummy0.tif"))
    assert "time" in out.dims
    assert out.sizes["time"] == 1
    assert out.coords["time"][0] == pd.Timestamp("2020-01-01T00:00:00")

def test_assign_time_skips_if_parse_none(loader, da_yx, monkeypatch):
    # Patch _parse_time to return None
    monkeypatch.setattr(loader, "_parse_time", lambda path: None)
    out = loader._assign_time(da_yx, Path("dummy.tif"))
    # Should remain unchanged, no time dimension
    assert "time" not in out.dims
    assert out.equals(da_yx)

def test_assign_time_multiple_calls(loader, da_yx):
    # Call twice, should assign time only once
    out = loader._assign_time(da_yx, Path("dummy1.tif"))
    out2 = loader._assign_time(out, Path("dummy2.tif"))
    # Time should remain a single-dim coordinate from the first assignment
    assert "time" in out2.dims
    assert out2.sizes["time"] == 1
    assert out2.coords["time"][0] == pd.Timestamp("2020-01-02")

@pytest.fixture
def fake_urls(tmp_path):
    """Create fake GeoTIFF-like files (empty but named)"""
    files = []
    for name in ["S1_VV.tif", "S1_VH.tif", "S1_mask.tif"]:
        f = tmp_path / name
        f.write_text("dummy")  # just to create a Path
        files.append(str(f))
    return files