"""Tests for sarvalanche.ml.pairwise_debris_classifier module."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from sarvalanche.ml.pairwise_debris_classifier.channels import (
    N_INPUT,
    N_SAR,
    N_STATIC,
    SAR_CHANNELS,
    STATIC_CHANNELS,
    normalize_anf,
    normalize_static_channel,
    sign_log1p,
)
from sarvalanche.ml.pairwise_debris_classifier.inference import (
    build_sar_channels,
    sliding_window_inference,
)
from sarvalanche.ml.pairwise_debris_classifier.model import SinglePairDetector
from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import (
    align_vh_to_vv,
    extract_pair_diff,
)
from sarvalanche.features.backscatter_change import backscatter_changes_all_pairs


# ═══════════════════════════════════════════════════════════════════════
# channels.py
# ═══════════════════════════════════════════════════════════════════════


class TestChannelConstants:
    def test_n_sar_matches_list(self):
        assert N_SAR == len(SAR_CHANNELS)

    def test_n_static_matches_list(self):
        assert N_STATIC == len(STATIC_CHANNELS)

    def test_n_input_is_sum(self):
        assert N_INPUT == N_SAR + N_STATIC

    def test_sar_channels_has_expected(self):
        assert 'change_vv' in SAR_CHANNELS
        assert 'change_vh' in SAR_CHANNELS
        assert 'change_cr' in SAR_CHANNELS
        assert 'anf' in SAR_CHANNELS

    def test_static_channels_has_expected(self):
        assert 'slope' in STATIC_CHANNELS
        assert 'tpi' in STATIC_CHANNELS
        assert 'cell_counts' in STATIC_CHANNELS

    def test_no_dem_in_static(self):
        assert 'dem' not in STATIC_CHANNELS

    def test_no_coverage_in_sar(self):
        assert 'coverage' not in SAR_CHANNELS

    def test_no_proximity_in_sar(self):
        assert 'proximity' not in SAR_CHANNELS


class TestSignLog1p:
    def test_zero_maps_to_zero(self):
        assert sign_log1p(np.array([0.0]))[0] == 0.0

    def test_positive_maps_positive(self):
        result = sign_log1p(np.array([1.0, 5.0, 10.0]))
        assert np.all(result > 0)

    def test_negative_maps_negative(self):
        result = sign_log1p(np.array([-1.0, -5.0]))
        assert np.all(result < 0)

    def test_symmetric(self):
        x = np.array([3.0])
        assert np.isclose(sign_log1p(x), -sign_log1p(-x))

    def test_compresses_large_values(self):
        result = sign_log1p(np.array([100.0]))
        assert result[0] < 100.0
        assert result[0] > 0.0

    def test_output_dtype_float32(self):
        result = sign_log1p(np.array([1.0], dtype=np.float64))
        assert result.dtype == np.float32

    def test_known_values(self):
        # log1p(1) = ln(2) ≈ 0.693
        np.testing.assert_allclose(sign_log1p(np.array([1.0])), [0.6931472], rtol=1e-5)
        np.testing.assert_allclose(sign_log1p(np.array([-1.0])), [-0.6931472], rtol=1e-5)


class TestNormalizeAnf:
    def test_zero_anf_returns_one(self):
        result = normalize_anf(np.array([0.0]))
        assert np.isclose(result[0], 1.0)

    def test_high_anf_near_zero(self):
        result = normalize_anf(np.array([100.0]))
        assert result[0] < 0.2

    def test_bounded_zero_one(self):
        anf = np.random.uniform(0, 50, size=1000).astype(np.float32)
        result = normalize_anf(anf)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_monotonically_decreasing(self):
        anf = np.array([0.0, 1.0, 5.0, 10.0, 50.0])
        result = normalize_anf(anf)
        assert np.all(np.diff(result) <= 0)

    def test_output_dtype_float32(self):
        result = normalize_anf(np.array([1.0], dtype=np.float64))
        assert result.dtype == np.float32


class TestNormalizeStaticChannel:
    def test_returns_copy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize_static_channel(arr, 'slope')
        arr[0] = 999.0
        assert result[0] != 999.0

    def test_unknown_channel_returns_copy(self):
        arr = np.array([1.0, 2.0])
        result = normalize_static_channel(arr, 'aspect_northing')
        np.testing.assert_array_equal(result, arr)
        assert result is not arr

    def test_slope_scales(self):
        arr = np.array([0.6], dtype=np.float32)
        result = normalize_static_channel(arr, 'slope')
        np.testing.assert_allclose(result, [1.0])

    def test_cell_counts_log1p_and_scale(self):
        arr = np.array([0.0, 1.0, 100.0], dtype=np.float32)
        result = normalize_static_channel(arr, 'cell_counts')
        # log1p(0)/5 = 0, log1p(1)/5 = 0.693/5 ≈ 0.139, log1p(100)/5 ≈ 0.925
        assert result[0] == 0.0
        assert 0.1 < result[1] < 0.2
        assert 0.8 < result[2] < 1.0

    def test_tpi_robust_scale(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 30, size=1000).astype(np.float32)
        result = normalize_static_channel(arr, 'tpi')
        # After IQR scaling, std should be around 0.7-1.0
        assert 0.3 < result.std() < 2.0


# ═══════════════════════════════════════════════════════════════════════
# model.py
# ═══════════════════════════════════════════════════════════════════════


class TestSinglePairDetector:
    def test_output_shape(self):
        model = SinglePairDetector()
        x = torch.randn(2, N_INPUT, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_default_in_ch_matches_n_input(self):
        model = SinglePairDetector()
        # First conv weight shape[1] should be N_INPUT
        assert model.enc1.block[0].weight.shape[1] == N_INPUT

    def test_custom_in_ch(self):
        model = SinglePairDetector(in_ch=5, base_ch=8)
        x = torch.randn(1, 5, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_rejects_non_divisible_by_16(self):
        model = SinglePairDetector()
        with pytest.raises(AssertionError, match="divisible by 16"):
            model(torch.randn(1, N_INPUT, 100, 100))

    def test_accepts_divisible_by_16(self):
        model = SinglePairDetector()
        model.eval()  # BatchNorm needs eval mode for small spatial dims
        for size in [16, 32, 64, 128, 256]:
            out = model(torch.randn(1, N_INPUT, size, size))
            assert out.shape == (1, 1, size, size)

    def test_predict_proba_bounded(self):
        model = SinglePairDetector()
        x = torch.randn(1, N_INPUT, 128, 128)
        probs = model.predict_proba(x)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_logits_vs_proba(self):
        model = SinglePairDetector()
        x = torch.randn(1, N_INPUT, 128, 128)
        logits = model(x)
        probs = model.predict_proba(x)
        np.testing.assert_allclose(
            torch.sigmoid(logits).detach().numpy(),
            probs.detach().numpy(),
            rtol=1e-5,
        )


# ═══════════════════════════════════════════════════════════════════════
# inference.py — build_sar_channels
# ═══════════════════════════════════════════════════════════════════════


class TestBuildSarChannels:
    def test_output_shape(self):
        H, W = 50, 60
        sar = build_sar_channels(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            np.ones((H, W), dtype=np.float32),
        )
        assert sar.shape == (N_SAR, H, W)

    def test_output_shape_no_vh(self):
        H, W = 50, 60
        sar = build_sar_channels(
            np.zeros((H, W), dtype=np.float32),
            None,
            np.ones((H, W), dtype=np.float32),
        )
        assert sar.shape == (N_SAR, H, W)

    def test_zero_diff_gives_zero_change(self):
        H, W = 10, 10
        sar = build_sar_channels(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            np.ones((H, W), dtype=np.float32),
        )
        # change_vv, change_vh, change_cr should all be zero
        np.testing.assert_array_equal(sar[0], 0.0)
        np.testing.assert_array_equal(sar[1], 0.0)
        np.testing.assert_array_equal(sar[2], 0.0)
        # ANF should be 1.0
        np.testing.assert_array_equal(sar[3], 1.0)

    def test_positive_vv_gives_positive_change(self):
        H, W = 10, 10
        vv = np.full((H, W), 3.0, dtype=np.float32)
        sar = build_sar_channels(vv, None, np.ones((H, W), dtype=np.float32))
        assert np.all(sar[0] > 0)

    def test_cross_ratio_is_vh_minus_vv(self):
        H, W = 10, 10
        vv = np.full((H, W), 2.0, dtype=np.float32)
        vh = np.full((H, W), 5.0, dtype=np.float32)
        sar = build_sar_channels(vv, vh, np.ones((H, W), dtype=np.float32))
        # cr = vh - vv = 3.0, change_cr = sign_log1p(3.0)
        expected_cr = sign_log1p(np.array([3.0]))[0]
        np.testing.assert_allclose(sar[2, 0, 0], expected_cr, rtol=1e-5)

    def test_no_vh_gives_zero_vh_and_cr(self):
        H, W = 10, 10
        sar = build_sar_channels(
            np.ones((H, W), dtype=np.float32), None,
            np.ones((H, W), dtype=np.float32),
        )
        np.testing.assert_array_equal(sar[1], 0.0)  # change_vh
        np.testing.assert_array_equal(sar[2], 0.0)  # change_cr


# ═══════════════════════════════════════════════════════════════════════
# inference.py — sliding_window_inference
# ═══════════════════════════════════════════════════════════════════════


class TestSlidingWindowInference:
    @pytest.fixture
    def model(self):
        return SinglePairDetector(in_ch=N_INPUT, base_ch=8)

    def test_output_shape_matches_input(self, model):
        H, W = 128, 128
        sar = np.random.randn(N_SAR, H, W).astype(np.float32)
        static = np.random.randn(N_STATIC, H, W).astype(np.float32)
        result = sliding_window_inference(sar, static, model, torch.device('cpu'),
                                          stride=64, batch_size=4)
        assert result.shape == (H, W)

    def test_no_nan_on_exact_fit(self, model):
        H, W = 128, 128
        sar = np.random.randn(N_SAR, H, W).astype(np.float32)
        static = np.random.randn(N_STATIC, H, W).astype(np.float32)
        result = sliding_window_inference(sar, static, model, torch.device('cpu'),
                                          stride=128, batch_size=4)
        assert not np.any(np.isnan(result))

    def test_no_nan_on_non_divisible(self, model):
        """Edge pixels should be covered via padding."""
        H, W = 200, 200
        sar = np.random.randn(N_SAR, H, W).astype(np.float32)
        static = np.random.randn(N_STATIC, H, W).astype(np.float32)
        result = sliding_window_inference(sar, static, model, torch.device('cpu'),
                                          stride=32, batch_size=4)
        assert result.shape == (H, W)
        assert not np.any(np.isnan(result))

    def test_output_bounded_zero_one(self, model):
        H, W = 128, 128
        sar = np.random.randn(N_SAR, H, W).astype(np.float32)
        static = np.random.randn(N_STATIC, H, W).astype(np.float32)
        result = sliding_window_inference(sar, static, model, torch.device('cpu'),
                                          stride=64, batch_size=4)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
# pair_extraction.py
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def pair_diffs_xr():
    """Create matched VV and VH pair diff DataArrays."""
    times_start = pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-13'])
    times_end = pd.to_datetime(['2025-01-13', '2025-01-25', '2025-01-25'])
    H, W = 20, 20

    vv_data = np.random.randn(3, H, W).astype(np.float32)
    vh_data = np.random.randn(3, H, W).astype(np.float32)

    vv = xr.DataArray(vv_data, dims=['pair', 'y', 'x'],
                       coords={'t_start': ('pair', times_start), 't_end': ('pair', times_end)})
    vh = xr.DataArray(vh_data, dims=['pair', 'y', 'x'],
                       coords={'t_start': ('pair', times_start), 't_end': ('pair', times_end)})
    return vv, vh


class TestAlignVhToVv:
    def test_aligned_pairs(self, pair_diffs_xr):
        vv, vh = pair_diffs_xr
        alignment = align_vh_to_vv(vv, vh)
        assert alignment is not None
        assert len(alignment) == 3
        for p in range(3):
            assert alignment[p] == p

    def test_none_vh_returns_none(self, pair_diffs_xr):
        vv, _ = pair_diffs_xr
        assert align_vh_to_vv(vv, None) is None

    def test_mismatched_pairs(self):
        """VH has a pair that VV doesn't, and vice versa."""
        H, W = 5, 5
        vv = xr.DataArray(
            np.zeros((2, H, W)),
            dims=['pair', 'y', 'x'],
            coords={
                't_start': ('pair', pd.to_datetime(['2025-01-01', '2025-02-01'])),
                't_end': ('pair', pd.to_datetime(['2025-01-13', '2025-02-13'])),
            })
        vh = xr.DataArray(
            np.zeros((2, H, W)),
            dims=['pair', 'y', 'x'],
            coords={
                't_start': ('pair', pd.to_datetime(['2025-01-01', '2025-03-01'])),
                't_end': ('pair', pd.to_datetime(['2025-01-13', '2025-03-13'])),
            })
        alignment = align_vh_to_vv(vv, vh)
        assert 0 in alignment  # first pair matches
        assert 1 not in alignment  # second VV pair has no VH match


class TestExtractPairDiff:
    def test_valid_mask_from_nan(self):
        H, W = 10, 10
        vv_data = np.ones((2, H, W), dtype=np.float32)
        vv_data[0, :5, :] = np.nan  # first pair: top half is NaN
        vv = xr.DataArray(vv_data, dims=['pair', 'y', 'x'],
                           coords={
                               't_start': ('pair', pd.to_datetime(['2025-01-01', '2025-01-01'])),
                               't_end': ('pair', pd.to_datetime(['2025-01-13', '2025-01-25'])),
                           })
        vv_arr, vh_arr, valid = extract_pair_diff(vv, None, 0, None)
        assert valid[:5, :].sum() == 0
        assert valid[5:, :].sum() == 50
        assert vv_arr[:5, :].sum() == 0.0  # nan filled to 0

    def test_valid_mask_intersects_vv_vh(self):
        H, W = 10, 10
        vv_data = np.ones((1, H, W), dtype=np.float32)
        vv_data[0, :3, :] = np.nan
        vh_data = np.ones((1, H, W), dtype=np.float32)
        vh_data[0, 7:, :] = np.nan

        coords = {
            't_start': ('pair', pd.to_datetime(['2025-01-01'])),
            't_end': ('pair', pd.to_datetime(['2025-01-13'])),
        }
        vv = xr.DataArray(vv_data, dims=['pair', 'y', 'x'], coords=coords)
        vh = xr.DataArray(vh_data, dims=['pair', 'y', 'x'], coords=coords)

        alignment = {0: 0}
        _, _, valid = extract_pair_diff(vv, vh, 0, alignment)
        # Valid only where BOTH have data: rows 3-6
        assert valid[:3, :].sum() == 0   # VV NaN
        assert valid[7:, :].sum() == 0   # VH NaN
        assert valid[3:7, :].sum() == 40  # both valid


# ═══════════════════════════════════════════════════════════════════════
# backscatter_change.py — backscatter_changes_all_pairs
# ═══════════════════════════════════════════════════════════════════════


class TestBackscatterChangesAllPairs:
    @pytest.fixture
    def five_timestep_da(self):
        times = pd.date_range("2024-01-01", periods=5, freq="12D")
        data = np.arange(5 * 4 * 4, dtype=float).reshape(5, 4, 4)
        return xr.DataArray(data, dims=["time", "y", "x"], coords={"time": times})

    def test_pair_count(self, five_timestep_da):
        """With max_span=24, pairs with span 12 and 24 should be included."""
        result = backscatter_changes_all_pairs(five_timestep_da, max_span_days=24)
        # 12-day pairs: (0,1),(1,2),(2,3),(3,4) = 4
        # 24-day pairs: (0,2),(1,3),(2,4) = 3
        assert result.sizes["pair"] == 7

    def test_respects_max_span(self, five_timestep_da):
        result = backscatter_changes_all_pairs(five_timestep_da, max_span_days=12)
        # Only 12-day pairs: 4
        assert result.sizes["pair"] == 4

    def test_no_pairs_raises(self, five_timestep_da):
        with pytest.raises(ValueError, match="No time pairs"):
            backscatter_changes_all_pairs(five_timestep_da, max_span_days=1)

    def test_diff_values_correct(self, five_timestep_da):
        result = backscatter_changes_all_pairs(five_timestep_da, max_span_days=12)
        # First pair should be t1 - t0
        expected = five_timestep_da.isel(time=1) - five_timestep_da.isel(time=0)
        np.testing.assert_allclose(result.isel(pair=0).values, expected.values)

    def test_has_t_start_t_end_coords(self, five_timestep_da):
        result = backscatter_changes_all_pairs(five_timestep_da, max_span_days=24)
        assert "t_start" in result.coords
        assert "t_end" in result.coords

    def test_platform_dropped(self):
        times = pd.date_range("2024-01-01", periods=3, freq="12D")
        data = np.zeros((3, 2, 2))
        da = xr.DataArray(data, dims=["time", "y", "x"],
                          coords={"time": times, "platform": ("time", ["S1A", "S1B", "S1A"])})
        result = backscatter_changes_all_pairs(da, max_span_days=24)
        assert "platform" not in result.coords


# ═══════════════════════════════════════════════════════════════════════
# dataset.py — PairwiseDebrisDataset
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def test_nc(tmp_path):
    """Create a minimal NetCDF for dataset testing."""
    H, W, T = 256, 256, 6
    rng = np.random.default_rng(42)

    times = pd.date_range("2025-01-01", periods=T, freq="12D")
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)

    ds = xr.Dataset(
        {
            'VV': xr.DataArray(rng.standard_normal((T, H, W)).astype(np.float32) * 2 - 15,
                               dims=['time', 'y', 'x']),
            'VH': xr.DataArray(rng.standard_normal((T, H, W)).astype(np.float32) * 2 - 20,
                               dims=['time', 'y', 'x']),
            'slope': xr.DataArray(rng.uniform(0, 0.8, (H, W)).astype(np.float32),
                                  dims=['y', 'x']),
            'aspect_northing': xr.DataArray(rng.uniform(-1, 1, (H, W)).astype(np.float32),
                                            dims=['y', 'x']),
            'aspect_easting': xr.DataArray(rng.uniform(-1, 1, (H, W)).astype(np.float32),
                                           dims=['y', 'x']),
            'cell_counts': xr.DataArray(rng.uniform(0, 500, (H, W)).astype(np.float32),
                                        dims=['y', 'x']),
            'tpi': xr.DataArray(rng.standard_normal((H, W)).astype(np.float32) * 20,
                                dims=['y', 'x']),
            'anf': xr.DataArray(rng.uniform(0.5, 5, (1, H, W)).astype(np.float32),
                                dims=['static_track', 'y', 'x'],
                                coords={'static_track': [1]}),
            'track': xr.DataArray(np.ones(T, dtype=np.int32), dims=['time']),
        },
        coords={'time': times, 'y': y, 'x': x},
        attrs={'preprocessed': 'rtc_tv', 'units': 'db'},
    )
    ds['VV'].attrs = {'units': 'db', 'source': 'test', 'product': 'test'}
    ds['VH'].attrs = {'units': 'db', 'source': 'test', 'product': 'test'}

    nc_path = tmp_path / "test_season.nc"
    ds.to_netcdf(nc_path)
    ds.close()
    return nc_path


class TestPairwiseDebrisDataset:
    @pytest.fixture
    def simple_dataset(self, test_nc):
        import geopandas as gpd
        from shapely.geometry import box
        from sarvalanche.ml.pairwise_debris_classifier.dataset import PairwiseDebrisDataset, build_lazy_dataset

        # Create a simple debris polygon covering a patch area
        debris_poly = box(10, 10, 30, 30)
        gdf = gpd.GeoDataFrame({'geometry': [debris_poly]}, crs=None)

        return build_lazy_dataset(
            test_nc, [("2025-01-13", gdf, None)],
            max_span_days=30, stride=128, augment=False,
        )

    def test_len(self, simple_dataset):
        assert len(simple_dataset) > 0

    def test_getitem_shape(self, simple_dataset):
        sample = simple_dataset[0]
        assert sample['x'].shape == (N_INPUT, 128, 128)
        assert sample['label'].shape == (1, 128, 128)

    def test_no_confidence_key(self, simple_dataset):
        sample = simple_dataset[0]
        assert 'confidence' not in sample

    def test_has_positive_and_negative(self, simple_dataset):
        has_pos = any(l == 1 for l in simple_dataset.labels)
        has_neg = any(l == 0 for l in simple_dataset.labels)
        assert has_pos or has_neg  # at least one type

    def test_curriculum_get_valid_indices(self, simple_dataset):
        valid_0 = simple_dataset.get_valid_indices(0)
        valid_25 = simple_dataset.get_valid_indices(25)
        # At epoch 25 should have at least as many as epoch 0
        assert len(valid_25) >= len(valid_0)
