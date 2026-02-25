import numpy as np
import pytest
import xarray as xr

from sarvalanche.masks.size_filter import filter_pixel_groups


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_group_mask():
    """
    5×5 boolean mask with two connected components:
      - A 4-pixel group (top-left 2×2 block)
      - A 1-pixel group (bottom-right corner)
    """
    data = np.zeros((5, 5), dtype=bool)
    data[0:2, 0:2] = True   # 4-pixel group
    data[4, 4] = True        # 1-pixel group
    return xr.DataArray(data, dims=["y", "x"], name="mask")


# ---------------------------------------------------------------------------
# min_size filtering
# ---------------------------------------------------------------------------

def test_min_size_removes_small_group(two_group_mask):
    result = filter_pixel_groups(two_group_mask, min_size=2)
    assert result.values[4, 4] == False
    assert result.values[0, 0] == True


def test_min_size_1_keeps_all_pixels(two_group_mask):
    result = filter_pixel_groups(two_group_mask, min_size=1)
    np.testing.assert_array_equal(result.values, two_group_mask.values)


# ---------------------------------------------------------------------------
# max_size filtering
# ---------------------------------------------------------------------------

def test_max_size_removes_large_group(two_group_mask):
    result = filter_pixel_groups(two_group_mask, max_size=2)
    # 4-pixel group removed, 1-pixel group kept
    assert result.values[0, 0] == False
    assert result.values[4, 4] == True


# ---------------------------------------------------------------------------
# Combined min and max
# ---------------------------------------------------------------------------

def test_min_and_max_removes_both_groups(two_group_mask):
    # min_size=2 removes 1-pixel, max_size=3 removes 4-pixel → all False
    result = filter_pixel_groups(two_group_mask, min_size=2, max_size=3)
    assert not result.values.any()


# ---------------------------------------------------------------------------
# return_nlabels
# ---------------------------------------------------------------------------

def test_return_nlabels_false_returns_dataarray(two_group_mask):
    result = filter_pixel_groups(two_group_mask, min_size=1)
    assert isinstance(result, xr.DataArray)


def test_return_nlabels_true_returns_tuple(two_group_mask):
    out = filter_pixel_groups(two_group_mask, min_size=2, return_nlabels=True)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_return_nlabels_count_after_filtering(two_group_mask):
    _, n = filter_pixel_groups(two_group_mask, min_size=2, return_nlabels=True)
    # Only the 4-pixel group survives min_size=2
    assert n == 1


# ---------------------------------------------------------------------------
# Shape and metadata preserved
# ---------------------------------------------------------------------------

def test_output_dims_preserved(two_group_mask):
    result = filter_pixel_groups(two_group_mask)
    assert result.dims == two_group_mask.dims


def test_output_sizes_preserved(two_group_mask):
    result = filter_pixel_groups(two_group_mask)
    assert result.sizes == two_group_mask.sizes


def test_output_name_contains_filtered(two_group_mask):
    result = filter_pixel_groups(two_group_mask)
    assert "filtered" in result.name


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_mask_stays_empty():
    data = np.zeros((4, 4), dtype=bool)
    da = xr.DataArray(data, dims=["y", "x"], name="m")
    result = filter_pixel_groups(da, min_size=1)
    assert not result.values.any()


def test_single_pixel_group_kept_when_min_size_1():
    data = np.zeros((3, 3), dtype=bool)
    data[1, 1] = True
    da = xr.DataArray(data, dims=["y", "x"], name="m")
    result = filter_pixel_groups(da, min_size=1)
    assert result.values[1, 1] == True
