import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from sarvalanche.ml.track_features import (
    PATCH_CHANNELS,
    N_PATCH_CHANNELS,
    STATIC_FEATURE_VARS,
    aggregate_seg_features,
    extract_track_features,
    extract_track_patch,
    extract_track_patch_with_target,
    _pixel_max_da,
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
            'p_pixelwise': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32), dims=['y', 'x'], coords=coords,
            ),
            'aspect': xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32) * 2 * np.pi, dims=['y', 'x'], coords=coords,
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


# ── _pixel_max_da ───────────────────────────────────────────────────────────


def test_pixel_max_da_basic():
    """Should return pixel-wise max across variables."""
    ds = _make_ds(10, 10)
    ds['a'] = ds['fcf'] * 0 + 1.0
    ds['b'] = ds['fcf'] * 0 + 2.0
    ds['a'] = ds['a'].rio.write_crs('EPSG:32611')
    ds['b'] = ds['b'].rio.write_crs('EPSG:32611')
    result = _pixel_max_da(ds, ['a', 'b'])
    assert np.allclose(result.values, 2.0)


def test_pixel_max_da_empty():
    """Empty var list should return None."""
    ds = _make_ds(10, 10)
    assert _pixel_max_da(ds, []) is None


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


# ── extract_track_patch ─────────────────────────────────────────────────────


def test_extract_track_patch_shape():
    """Output shape should be (N_PATCH_CHANNELS, size, size)."""
    ds = _make_ds()
    row = _make_track_row(ds)
    patch = extract_track_patch(row, ds, size=32)
    assert patch.shape == (N_PATCH_CHANNELS, 32, 32)
    assert patch.dtype == np.float32


def test_extract_track_patch_coordinate_channels():
    """Northing and easting channels should have correct ranges."""
    ds = _make_ds()
    row = _make_track_row(ds)
    patch = extract_track_patch(row, ds, size=64)
    # Channel 4: northing — top=+1, bottom=-1
    assert abs(patch[4, 0, 0] - 1.0) < 1e-5  # top-left
    assert abs(patch[4, -1, 0] - (-1.0)) < 1e-5  # bottom-left
    # Channel 5: easting — left=-1, right=+1
    assert abs(patch[5, 0, 0] - (-1.0)) < 1e-5  # top-left
    assert abs(patch[5, 0, -1] - 1.0) < 1e-5  # top-right


def test_extract_track_patch_track_mask():
    """Track mask channel should have values in {0, 1}."""
    ds = _make_ds()
    row = _make_track_row(ds)
    patch = extract_track_patch(row, ds, size=64)
    mask = patch[6]
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0  # should have some pixels inside track


def test_extract_track_patch_missing_vars():
    """If no data vars found, should return zeros."""
    ds = _make_ds()
    row = _make_track_row(ds)
    # Remove all patch data vars
    for v in ['distance_mahalanobis', 'p_empirical', 'fcf', 'slope']:
        del ds[v]
    patch = extract_track_patch(row, ds, size=32)
    assert np.allclose(patch, 0.0)


# ── extract_track_patch_with_target ─────────────────────────────────────────


def test_extract_track_patch_with_target_shapes():
    """Should return patch and target with correct shapes."""
    ds = _make_ds()
    row = _make_track_row(ds)
    patch, target = extract_track_patch_with_target(row, ds, size=32)
    assert patch.shape == (N_PATCH_CHANNELS, 32, 32)
    assert target.shape == (1, 32, 32)
    assert target.dtype == np.float32


def test_extract_track_patch_with_target_no_pixelwise():
    """Without p_pixelwise, target should be all zeros."""
    ds = _make_ds()
    del ds['p_pixelwise']
    row = _make_track_row(ds)
    patch, target = extract_track_patch_with_target(row, ds, size=32)
    assert np.allclose(target, 0.0)


def test_extract_track_patch_with_target_debris_shapes():
    """Debris shapes should be blended into target via max."""
    ds = _make_ds()
    # Zero out p_pixelwise so we can check that shapes add signal
    ds['p_pixelwise'].values[:] = 0.0
    row = _make_track_row(ds)
    # Create a debris shape covering the track center
    geom = row.geometry
    shapes_gdf = gpd.GeoDataFrame({'key': ['test'], 'geometry': [geom]}, crs='EPSG:32611')
    _, target_with = extract_track_patch_with_target(row, ds, size=32, debris_shapes=shapes_gdf)
    _, target_without = extract_track_patch_with_target(row, ds, size=32)
    # Target with shapes should have more nonzero pixels
    assert target_with.sum() >= target_without.sum()
