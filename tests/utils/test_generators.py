import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sarvalanche.utils.generators import iter_track_pol_combinations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_track_two_pol_ds():
    """Dataset with VV and VH, two tracks (1 and 2), two timesteps each."""
    times = pd.date_range("2024-01-01", periods=4, freq="D")
    track_vals = np.array([1, 1, 2, 2])

    rng = np.random.default_rng(42)
    vv = rng.random((4, 3, 3))
    vh = rng.random((4, 3, 3))

    return xr.Dataset(
        {
            "VV": xr.DataArray(vv, dims=["time", "y", "x"], coords={"time": times}),
            "VH": xr.DataArray(vh, dims=["time", "y", "x"], coords={"time": times}),
            "track": ("time", track_vals),
        }
    )


@pytest.fixture
def vv_only_ds():
    """Dataset with only VV, single track."""
    times = pd.date_range("2024-01-01", periods=2, freq="D")
    track_vals = np.array([1, 1])
    rng = np.random.default_rng(0)
    vv = rng.random((2, 3, 3))
    return xr.Dataset(
        {
            "VV": xr.DataArray(vv, dims=["time", "y", "x"], coords={"time": times}),
            "track": ("time", track_vals),
        }
    )


# ---------------------------------------------------------------------------
# Yield count
# ---------------------------------------------------------------------------

def test_yields_all_track_pol_combinations(two_track_two_pol_ds):
    results = list(iter_track_pol_combinations(two_track_two_pol_ds))
    # 2 tracks × 2 polarizations = 4 tuples
    assert len(results) == 4


def test_yields_correct_track_values(two_track_two_pol_ds):
    results = list(iter_track_pol_combinations(two_track_two_pol_ds))
    tracks = {r[0] for r in results}
    assert tracks == {1, 2}


def test_yields_correct_pol_values(two_track_two_pol_ds):
    results = list(iter_track_pol_combinations(two_track_two_pol_ds))
    pols = {r[1] for r in results}
    assert pols == {"VV", "VH"}


def test_yielded_da_is_dataarray(two_track_two_pol_ds):
    for _, _, da in iter_track_pol_combinations(two_track_two_pol_ds):
        assert isinstance(da, xr.DataArray)


def test_yielded_da_has_only_that_tracks_times(two_track_two_pol_ds):
    """Each yielded DataArray should contain only the time steps for that track."""
    for track, _, da in iter_track_pol_combinations(two_track_two_pol_ds):
        # 2 time steps per track in this fixture
        assert da.sizes["time"] == 2


# ---------------------------------------------------------------------------
# skip_missing=True (default)
# ---------------------------------------------------------------------------

def test_skip_missing_true_skips_absent_pol(vv_only_ds):
    results = list(
        iter_track_pol_combinations(vv_only_ds, polarizations=["VV", "VH"], skip_missing=True)
    )
    # Only VV is present → 1 result (1 track × 1 pol)
    assert len(results) == 1
    assert results[0][1] == "VV"


# ---------------------------------------------------------------------------
# skip_missing=False
# ---------------------------------------------------------------------------

def test_skip_missing_false_raises_on_absent_pol(vv_only_ds):
    with pytest.raises(KeyError, match="VH"):
        list(
            iter_track_pol_combinations(
                vv_only_ds, polarizations=["VV", "VH"], skip_missing=False
            )
        )


# ---------------------------------------------------------------------------
# Missing track variable
# ---------------------------------------------------------------------------

def test_missing_track_var_raises():
    ds = xr.Dataset({"VV": xr.DataArray([1.0], dims=["x"])})
    with pytest.raises(ValueError, match="track"):
        list(iter_track_pol_combinations(ds))


# ---------------------------------------------------------------------------
# Custom polarization list
# ---------------------------------------------------------------------------

def test_custom_single_polarization(two_track_two_pol_ds):
    results = list(
        iter_track_pol_combinations(two_track_two_pol_ds, polarizations=["VV"])
    )
    # 2 tracks × 1 pol = 2
    assert len(results) == 2
    assert all(r[1] == "VV" for r in results)
