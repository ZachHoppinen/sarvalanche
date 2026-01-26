import pytest
from pathlib import Path
import xarray as xr
import numpy as np
from shapely.geometry import box, Point
from sarvalanche.io.sources.DemSource import DemSource

# Mock py3dep
import py3dep

@pytest.fixture
def dummy_aoi():
    return box(-155.5, 19.9, -155.4, 20.0)

@pytest.fixture
def tmp_dem_file(tmp_path):
    # create a tiny raster
    data = np.ones((10, 10), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(10),
            "x": np.arange(10)
        },
        name="dem"
    )
    da = da.rio.write_crs("EPSG:4326")
    path = tmp_path / "dem.tif"
    da.rio.to_raster(path)
    return path

def test_load_py3dep(monkeypatch, dummy_aoi):
    # Patch py3dep.get_map to return a dummy DataArray
    dummy_da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"), name="dem")
    monkeypatch.setattr(py3dep, "get_map", lambda *args, **kwargs: dummy_da)

    src = DemSource(cache_dir='')
    dem = src.load(aoi=dummy_aoi, resolution=10, epsg=4326)

    assert isinstance(dem, xr.DataArray)
    assert dem.name == "dem"
    assert dem.attrs["sensor"] == "DEM"
    assert dem.attrs["source"] == "py3dep"
    assert "y" in dem.dims and "x" in dem.dims
    # validate canonical runs without error
    from sarvalanche.utils.validation import validate_canonical
    validate_canonical(dem)

def test_load_user_dem(tmp_dem_file):
    src = DemSource()
    dem = src.load(aoi=None, dem_path=tmp_dem_file, resolution=1, epsg=4326)

    assert isinstance(dem, xr.DataArray)
    assert dem.name == "dem"
    assert dem.attrs["sensor"] == "DEM"
    assert dem.attrs["source"] == "user"
    assert dem.shape[0] == 10 and dem.shape[1] == 10

def test_load_user_dem_not_found():
    src = DemSource()
    with pytest.raises(FileNotFoundError):
        src.load(aoi=None, dem_path="nonexistent_file.tif")
