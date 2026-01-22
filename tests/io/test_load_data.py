import numpy as np
import xarray as xr

import sarvalanche.io as io


def test_load_data_returns_canonical_dataset(
    mocker,
    canonical_sar_da,
    aoi,
):
    fake_urls = ["s1_001", "s1_002"]

    # Mock download step
    mocker.patch(
        "sarvalanche.io.load.download_product",
        return_value="/tmp/fake_file"
    )

    # Mock conversion step
    mocker.patch(
        "sarvalanche.io.load.convert_to_canonical",
        return_value=canonical_sar_da
    )

    ds = io.load_data(
        urls=fake_urls,
        aoi=aoi,
    )

    assert isinstance(ds, xr.Dataset)
    assert "sigma0" in ds
    assert set(ds.dims) == {"time", "y", "x"}
    assert ds.sizes["time"] == 2

def test_load_data_sorts_time_axis(mocker, canonical_sar_da, aoi):
    da1 = canonical_sar_da.assign_coords(
        time=["2024-01-10"]
    )
    da2 = canonical_sar_da.assign_coords(
        time=["2024-01-01"]
    )

    mocker.patch(
        "sarvalanche.io.load.download_product",
        return_value="/tmp/fake"
    )

    mocker.patch(
        "sarvalanche.io.load.convert_to_canonical",
        side_effect=[da1, da2]
    )

    ds = io.load_data(
        urls=["a", "b"],
        aoi=aoi,
    )

    times = ds.time.values
    assert np.all(times[:-1] <= times[1:])

def test_load_data_sets_crs(mocker, canonical_sar_da, aoi):
    canonical_sar_da = canonical_sar_da.rio.write_crs(None)

    mocker.patch(
        "sarvalanche.io.load.download_product",
        return_value="/tmp/fake"
    )
    mocker.patch(
        "sarvalanche.io.load.convert_to_canonical",
        return_value=canonical_sar_da
    )

    ds = io.load_data(
        urls=["a"],
        aoi=aoi,
    )

    assert ds.rio.crs is not None
