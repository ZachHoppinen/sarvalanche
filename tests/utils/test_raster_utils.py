"""Tests for sarvalanche.utils.raster_utils.combine_close_images."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sarvalanche.utils.raster_utils import combine_close_images, mosaic_group


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_da():
    """4 time steps: first two within 1 min (should merge), last two 12 days apart."""
    times = pd.to_datetime([
        "2024-01-01T12:00:00",
        "2024-01-01T12:00:30",  # 30s after first — same pass
        "2024-01-13T12:00:00",  # 12 days later
        "2024-01-25T12:00:00",  # 12 days later
    ])
    ny, nx = 10, 12
    rng = np.random.default_rng(42)
    data = rng.uniform(0.01, 0.5, (4, ny, nx)).astype("float32")
    # Make the two close frames complementary: first has NaN in right half,
    # second has NaN in left half
    data[0, :, nx // 2:] = np.nan
    data[1, :, :nx // 2] = np.nan

    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": np.arange(ny, dtype="float64"),
            "x": np.arange(nx, dtype="float64"),
        },
    )


@pytest.fixture
def no_merge_da():
    """3 time steps all well separated — nothing should merge."""
    times = pd.to_datetime([
        "2024-01-01T12:00:00",
        "2024-01-13T12:00:00",
        "2024-01-25T12:00:00",
    ])
    ny, nx = 8, 8
    data = np.ones((3, ny, nx), dtype="float32")
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": np.arange(ny, dtype="float64"),
            "x": np.arange(nx, dtype="float64"),
        },
    )


@pytest.fixture
def multi_merge_da():
    """6 times: two groups of 2 close + 2 singletons."""
    times = pd.to_datetime([
        "2024-01-01T12:00:00",
        "2024-01-01T12:00:30",  # merge with above
        "2024-01-13T12:00:00",
        "2024-01-25T12:00:00",
        "2024-01-25T12:01:00",  # merge with above
        "2024-02-06T12:00:00",
    ])
    ny, nx = 6, 6
    data = np.arange(6 * ny * nx, dtype="float32").reshape(6, ny, nx)
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": np.arange(ny, dtype="float64"),
            "x": np.arange(nx, dtype="float64"),
        },
    )


# ---------------------------------------------------------------------------
# Tests: mosaic_group
# ---------------------------------------------------------------------------

class TestMosaicGroup:
    def test_single_timestep_passthrough(self):
        """A single-timestep group should be returned unchanged."""
        data = np.ones((1, 5, 5), dtype="float32")
        da = xr.DataArray(
            data, dims=("time", "y", "x"),
            coords={"time": [pd.Timestamp("2024-01-01")],
                     "y": np.arange(5.0), "x": np.arange(5.0)},
        )
        result = mosaic_group(da)
        assert result.shape == (1, 5, 5)
        np.testing.assert_array_equal(result.values, data)

    def test_two_frames_combine_first(self):
        """Two overlapping frames: first takes priority, second fills NaN."""
        ny, nx = 4, 6
        a = np.full((ny, nx), 1.0, dtype="float32")
        b = np.full((ny, nx), 2.0, dtype="float32")
        a[:, nx // 2:] = np.nan  # a has right half NaN
        data = np.stack([a, b])[np.newaxis]  # just to build shape
        data = np.stack([a, b])  # (2, ny, nx)

        da = xr.DataArray(
            data, dims=("time", "y", "x"),
            coords={
                "time": pd.to_datetime(["2024-01-01T12:00:00", "2024-01-01T12:00:30"]),
                "y": np.arange(ny, dtype="float64"),
                "x": np.arange(nx, dtype="float64"),
            },
        )
        result = mosaic_group(da)
        assert result.shape == (1, ny, nx)
        # Left half from a (1.0), right half from b (2.0)
        np.testing.assert_array_equal(result.values[0, :, :nx // 2], 1.0)
        np.testing.assert_array_equal(result.values[0, :, nx // 2:], 2.0)

    def test_mean_time(self):
        """Merged time should be the mean of input times."""
        times = pd.to_datetime(["2024-01-01T12:00:00", "2024-01-01T12:01:00"])
        data = np.ones((2, 3, 3), dtype="float32")
        da = xr.DataArray(
            data, dims=("time", "y", "x"),
            coords={"time": times, "y": np.arange(3.0), "x": np.arange(3.0)},
        )
        result = mosaic_group(da)
        expected_time = times.mean()
        assert result.time.values[0] == expected_time


# ---------------------------------------------------------------------------
# Tests: combine_close_images
# ---------------------------------------------------------------------------

class TestCombineCloseImages:
    def test_merges_close_frames(self, simple_da):
        """Two frames within 2 min should be merged into one."""
        result = combine_close_images(simple_da)
        # 4 inputs -> 3 outputs (first two merged)
        assert result.sizes["time"] == 3

    def test_merged_frame_fills_nans(self, simple_da):
        """Merged frame should have no NaN (complementary halves)."""
        result = combine_close_images(simple_da)
        merged = result.isel(time=0)
        assert not np.any(np.isnan(merged.values))

    def test_no_merge_when_separated(self, no_merge_da):
        """Well-separated frames should pass through unchanged."""
        result = combine_close_images(no_merge_da)
        assert result.sizes["time"] == 3
        np.testing.assert_array_equal(result.values, no_merge_da.values)

    def test_multiple_merge_groups(self, multi_merge_da):
        """Two merge groups + 2 singletons -> 4 output times."""
        result = combine_close_images(multi_merge_da)
        # 6 inputs: (merge 2) + 1 + (merge 2) + 1 = 4
        assert result.sizes["time"] == 4

    def test_preserves_spatial_dims(self, simple_da):
        """Output y, x dims should match input."""
        result = combine_close_images(simple_da)
        assert result.sizes["y"] == simple_da.sizes["y"]
        assert result.sizes["x"] == simple_da.sizes["x"]

    def test_preserves_dtype(self, simple_da):
        """Output dtype should match input."""
        result = combine_close_images(simple_da)
        assert result.dtype == simple_da.dtype

    def test_time_ordering_preserved(self, simple_da):
        """Output times should be monotonically increasing."""
        result = combine_close_images(simple_da)
        times = pd.DatetimeIndex(result.time.values)
        assert times.is_monotonic_increasing

    def test_custom_tolerance(self):
        """Custom time_tol should change merge behavior."""
        times = pd.to_datetime([
            "2024-01-01T12:00:00",
            "2024-01-01T12:05:00",  # 5 min apart
            "2024-01-13T12:00:00",
        ])
        data = np.ones((3, 4, 4), dtype="float32")
        da = xr.DataArray(
            data, dims=("time", "y", "x"),
            coords={"time": times, "y": np.arange(4.0), "x": np.arange(4.0)},
        )
        # Default 2 min: should NOT merge
        result_default = combine_close_images(da)
        assert result_default.sizes["time"] == 3

        # 10 min tolerance: SHOULD merge
        result_wide = combine_close_images(da, time_tol=pd.Timedelta("10min"))
        assert result_wide.sizes["time"] == 2

    def test_dask_backed_input(self, simple_da):
        """Dask-backed input should produce correct results without full materialization."""
        chunked = simple_da.chunk({"time": 2, "y": 5, "x": 6})
        result = combine_close_images(chunked)
        # Same merge behavior as eager
        assert result.sizes["time"] == 3
        # Merged frame should fill NaNs
        merged = result.isel(time=0).values
        assert not np.any(np.isnan(merged))
