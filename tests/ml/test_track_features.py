import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from sarvalanche.ml.track_features import (
    STATIC_FEATURE_VARS,
    extract_track_features,
)
from sarvalanche.ml.track_patch_extraction import (
    PATCH_CHANNELS,
    N_PATCH_CHANNELS,
    TRACK_MASK_CHANNEL,
    _NORTHING_CH,
    _EASTING_CH,
    _PATCH_DATA_VARS,
    aggregate_seg_features,
    extract_context_patch,
    _normalize_channel,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_ds(nx=50, ny=50, crs='EPSG:32611'):
    """Create a minimal projected dataset with standard variables."""
    x = np.arange(nx) * 30.0 + 500000.0
    y = np.arange(ny) * -30.0 + 5000000.0
    coords = {'y': y, 'x': x}

    ds = xr.Dataset(
        {
            'distance_mahalanobis': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'p_empirical': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'fcf': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'slope': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * 0.8, dims=['y', 'x'], coords=coords,
            ),
            'dem': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * 3000, dims=['y', 'x'], coords=coords,
            ),
            'cell_counts': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * 100, dims=['y', 'x'], coords=coords,
            ),
            'combined_distance': xr.DataArray(
                (np.random.rand(ny, nx).astype(np.float32) - 0.5) * 6, dims=['y', 'x'], coords=coords,
            ),
            'd_empirical': xr.DataArray(
                (np.random.rand(ny, nx).astype(np.float32) - 0.5) * 4, dims=['y', 'x'], coords=coords,
            ),
            'p_pixelwise': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'aspect': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * 2 * np.pi, dims=['y', 'x'], coords=coords,
            ),
            'release_zones': xr.DataArray(
                (np.random.rand(ny, nx) > 0.8).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'runout_angle': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * np.pi, dims=['y', 'x'], coords=coords,
            ),
        }
    )
    for v in ds.data_vars:
        ds[v] = ds[v].rio.write_crs(crs)
        ds[v].rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    return ds


def _make_track_row(ds):
    """Create a GeoSeries row with a box geometry inside the dataset extent."""
    x = ds.x.values
    y = ds.y.values
    cx, cy = float(x[len(x) // 2]), float(y[len(y) // 2])
    geom = box(cx - 200, cy - 200, cx + 200, cy + 200)
    gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:32611')
    return gdf.iloc[0]


# ── aggregate_seg_features ──────────────────────────────────────────────────


def test_aggregate_seg_features_basic():
    """Should return all expected keys with correct values."""
    seg_map = np.array([[0.8, 0.6], [0.3, 0.9]], dtype=np.float32)
    track_mask = np.ones((2, 2), dtype=np.float32)
    result = aggregate_seg_features(seg_map, track_mask)
    assert set(result.keys()) == {
        'seg_mean', 'seg_max', 'seg_p75', 'seg_p90', 'seg_p95', 'seg_frac_above_05',
    }
    assert abs(result['seg_mean'] - np.mean([0.8, 0.6, 0.3, 0.9])) < 1e-5
    assert abs(result['seg_max'] - 0.9) < 1e-5
    assert abs(result['seg_frac_above_05'] - 0.75) < 1e-5  # 3/4 above 0.5


def test_aggregate_seg_features_partial_mask():
    """Only pixels inside the mask should be aggregated."""
    seg_map = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    track_mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = aggregate_seg_features(seg_map, track_mask)
    assert abs(result['seg_mean'] - 0.9) < 1e-5


def test_aggregate_seg_features_empty_mask():
    """Empty mask should return NaN for all features."""
    seg_map = np.ones((4, 4), dtype=np.float32)
    track_mask = np.zeros((4, 4), dtype=np.float32)
    result = aggregate_seg_features(seg_map, track_mask)
    for v in result.values():
        assert np.isnan(v)


def test_aggregate_seg_features_no_mask():
    """When track_mask=None, should aggregate over all pixels."""
    seg_map = np.array([[0.8, 0.6], [0.3, 0.9]], dtype=np.float32)
    result = aggregate_seg_features(seg_map)
    assert abs(result['seg_mean'] - np.mean([0.8, 0.6, 0.3, 0.9])) < 1e-5
    assert abs(result['seg_max'] - 0.9) < 1e-5


# ── extract_track_features ──────────────────────────────────────────────────


def test_extract_track_features_keys():
    """Should return features for all static vars."""
    ds = _make_ds()
    row = _make_track_row(ds)
    feats = extract_track_features(row, ds)
    assert isinstance(feats, dict)
    # Should have area_pixels and stats for each static var
    assert 'area_pixels' in feats
    for var in STATIC_FEATURE_VARS:
        for stat in ['mean', 'max', 'std', 'p75', 'p90']:
            assert f'{var}_{stat}' in feats, f"Missing {var}_{stat}"


def test_extract_track_features_values_finite():
    """Feature values should be finite for a valid geometry."""
    ds = _make_ds()
    row = _make_track_row(ds)
    feats = extract_track_features(row, ds)
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_extract_track_features_missing_var():
    """Missing dataset variables should be skipped gracefully."""
    ds = _make_ds()
    del ds['cell_counts']
    row = _make_track_row(ds)
    feats = extract_track_features(row, ds)
    # cell_counts stats should be absent, but other vars should be present
    assert 'fcf_mean' in feats
    assert 'cell_counts_mean' not in feats


# ── _normalize_channel ─────────────────────────────────────────────────────


def test_channel_normalization():
    """Normalization should bring channels into reasonable ranges."""
    # cell_counts: log1p then /5
    arr = np.array([0, 1, 100, 1000], dtype=np.float32)
    out = _normalize_channel(arr, 'cell_counts')
    assert out[0] == pytest.approx(0.0)  # log1p(0) = 0
    assert out[1] == pytest.approx(np.log1p(1.0) / 5.0)
    assert out[3] < 1.5  # log1p(1000)/5 ≈ 1.38

    # combined_distance: /5
    arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    out = _normalize_channel(arr, 'combined_distance')
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])

    # slope: /0.6
    arr = np.array([0.0, 0.3, 0.6], dtype=np.float32)
    out = _normalize_channel(arr, 'slope')
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    # fcf: /100
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    out = _normalize_channel(arr, 'fcf')
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
