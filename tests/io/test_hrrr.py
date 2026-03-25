"""Tests for sarvalanche.io.hrrr — HRRR temperature fetching and processing."""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from sarvalanche.io.hrrr import (
    _projected_to_latlon,
    build_kdtree,
    detect_hrrr_model,
    fetch_t2m_batch,
    get_hrrr_for_dataset,
    lapse_rate_correct,
    nearest_cycle_hour,
    resample_with_tree,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def projected_ds():
    """Small SAR-like dataset in EPSG:32606 (UTM zone 6N — Alaska)."""
    ny, nx, nt = 20, 25, 3
    y = np.linspace(6_800_000, 6_800_600, ny)
    x = np.linspace(350_000, 350_750, nx)
    times = pd.date_range('2025-01-15 06:30', periods=nt, freq='12h')

    rng = np.random.default_rng(42)
    ds = xr.Dataset(
        {
            'VV': (['time', 'y', 'x'], rng.random((nt, ny, nx), dtype=np.float32)),
            'dem': (['y', 'x'], rng.uniform(500, 2000, (ny, nx)).astype(np.float32)),
        },
        coords={'time': times, 'y': y, 'x': x},
    )
    ds = ds.rio.write_crs('EPSG:32606')
    return ds


@pytest.fixture
def conus_ds():
    """Small dataset in EPSG:32611 (UTM zone 11N — Idaho/CONUS)."""
    ny, nx, nt = 10, 10, 2
    y = np.linspace(4_800_000, 4_800_300, ny)
    x = np.linspace(600_000, 600_300, nx)
    times = pd.date_range('2025-02-01 14:00', periods=nt, freq='12h')

    rng = np.random.default_rng(99)
    ds = xr.Dataset(
        {
            'VV': (['time', 'y', 'x'], rng.random((nt, ny, nx), dtype=np.float32)),
            'dem': (['y', 'x'], rng.uniform(1500, 3000, (ny, nx)).astype(np.float32)),
        },
        coords={'time': times, 'y': y, 'x': x},
    )
    ds = ds.rio.write_crs('EPSG:32611')
    return ds


@pytest.fixture
def latlon_ds():
    """Small dataset in EPSG:4326 (geographic)."""
    ny, nx, nt = 10, 10, 2
    y = np.linspace(61.0, 61.1, ny)
    x = np.linspace(-149.5, -149.4, nx)
    times = pd.date_range('2025-01-20 06:00', periods=nt, freq='12h')

    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {
            'VV': (['time', 'y', 'x'], rng.random((nt, ny, nx), dtype=np.float32)),
            'dem': (['y', 'x'], rng.uniform(300, 1500, (ny, nx)).astype(np.float32)),
        },
        coords={'time': times, 'y': y, 'x': x},
    )
    ds = ds.rio.write_crs('EPSG:4326')
    return ds


# ---------------------------------------------------------------------------
# nearest_cycle_hour
# ---------------------------------------------------------------------------


class TestNearestCycleHour:
    def test_exact_match_hrrrak(self):
        ts = pd.Timestamp('2025-01-15 06:00')
        assert nearest_cycle_hour(ts, model='hrrrak') == 6

    def test_round_down_hrrrak(self):
        ts = pd.Timestamp('2025-01-15 07:00')
        assert nearest_cycle_hour(ts, model='hrrrak') == 6

    def test_round_up_hrrrak(self):
        ts = pd.Timestamp('2025-01-15 08:00')
        assert nearest_cycle_hour(ts, model='hrrrak') == 9

    def test_midpoint_hrrrak(self):
        # 07:30 is equidistant between 6 and 9 — should pick 6 (first argmin)
        ts = pd.Timestamp('2025-01-15 07:30')
        result = nearest_cycle_hour(ts, model='hrrrak')
        assert result in (6, 9)

    def test_wrap_around_hrrrak(self):
        # 22:30 is closer to 21 (1.5h) than 0 (1.5h) — should pick 21 or 0
        ts = pd.Timestamp('2025-01-15 22:30')
        result = nearest_cycle_hour(ts, model='hrrrak')
        assert result in (21, 0)

    def test_near_midnight_hrrrak(self):
        # 23:00 is 2h from 21 and 1h from 0 — should pick 0
        ts = pd.Timestamp('2025-01-15 23:00')
        assert nearest_cycle_hour(ts, model='hrrrak') == 0

    def test_exact_match_conus(self):
        ts = pd.Timestamp('2025-01-15 14:00')
        assert nearest_cycle_hour(ts, model='hrrr') == 14

    def test_round_conus(self):
        ts = pd.Timestamp('2025-01-15 14:40')
        assert nearest_cycle_hour(ts, model='hrrr') == 15

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match='Unknown HRRR model'):
            nearest_cycle_hour(pd.Timestamp('2025-01-15'), model='bad_model')


# ---------------------------------------------------------------------------
# detect_hrrr_model
# ---------------------------------------------------------------------------


class TestDetectHrrrModel:
    def test_alaska_projected(self, projected_ds):
        assert detect_hrrr_model(projected_ds) == 'hrrrak'

    def test_conus_projected(self, conus_ds):
        assert detect_hrrr_model(conus_ds) == 'hrrr'

    def test_alaska_geographic(self, latlon_ds):
        assert detect_hrrr_model(latlon_ds) == 'hrrrak'

    def test_no_crs_raises(self):
        ds = xr.Dataset({'dem': (['y', 'x'], np.zeros((2, 2)))}, coords={'y': [0, 1], 'x': [0, 1]})
        with pytest.raises(ValueError, match='CRS'):
            detect_hrrr_model(ds)


# ---------------------------------------------------------------------------
# lapse_rate_correct
# ---------------------------------------------------------------------------


class TestLapseRateCorrect:
    def test_flat_dem_no_change(self):
        """Uniform DEM → smoothed == raw → zero correction."""
        t2m = np.full((10, 10), -5.0, dtype=np.float32)
        dem = np.full((10, 10), 1000.0, dtype=np.float32)
        result = lapse_rate_correct(t2m, dem, smooth_pixels=3)
        np.testing.assert_allclose(result, t2m, atol=0.01)

    def test_higher_pixel_is_colder(self):
        """A pixel higher than its surroundings should be adjusted colder."""
        dem = np.full((50, 50), 1000.0, dtype=np.float32)
        dem[25, 25] = 2000.0  # 1000m above surroundings
        t2m = np.zeros((50, 50), dtype=np.float32)

        result = lapse_rate_correct(t2m, dem, smooth_pixels=20)
        # The peak pixel should be colder (LAPSE_RATE is negative)
        assert result[25, 25] < result[0, 0]

    def test_nan_dem_treated_as_zero(self):
        """NaN in DEM should not propagate to output."""
        t2m = np.full((5, 5), 0.0, dtype=np.float32)
        dem = np.full((5, 5), 500.0, dtype=np.float32)
        dem[2, 2] = np.nan
        result = lapse_rate_correct(t2m, dem, smooth_pixels=3)
        assert np.all(np.isfinite(result))

    def test_output_dtype(self):
        t2m = np.zeros((5, 5))
        dem = np.ones((5, 5)) * 1000
        result = lapse_rate_correct(t2m, dem)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# KD-tree build + resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_identity_resample(self):
        """Resampling to the same grid should return the original values."""
        lat = np.array([[60.0, 60.0], [61.0, 61.0]])
        lon = np.array([[-150.0, -149.0], [-150.0, -149.0]])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        tree = build_kdtree(lat, lon)
        result = resample_with_tree(tree, values.ravel(), lat, lon)
        np.testing.assert_array_equal(result, values)

    def test_nearest_neighbour(self):
        """Query point closest to one source should get that value."""
        lat = np.array([[0.0, 0.0], [1.0, 1.0]])
        lon = np.array([[0.0, 1.0], [0.0, 1.0]])
        values = np.array([[10.0, 20.0], [30.0, 40.0]])

        tree = build_kdtree(lat, lon)

        # Target point very close to (1, 1) → value 40
        t_lat = np.array([[0.99]])
        t_lon = np.array([[0.99]])
        result = resample_with_tree(tree, values.ravel(), t_lat, t_lon)
        assert result[0, 0] == 40.0

    def test_output_shape(self):
        """Output shape should match target grid."""
        lat = np.ones((3, 3)) * 60
        lon = np.ones((3, 3)) * -150
        tree = build_kdtree(lat, lon)

        t_lat = np.zeros((5, 7))
        t_lon = np.zeros((5, 7))
        result = resample_with_tree(tree, np.ones(9), t_lat, t_lon)
        assert result.shape == (5, 7)


# ---------------------------------------------------------------------------
# _projected_to_latlon
# ---------------------------------------------------------------------------


class TestProjectedToLatlon:
    def test_geographic_passthrough(self):
        """EPSG:4326 should just meshgrid y/x as lat/lon."""
        y = np.array([60.0, 61.0])
        x = np.array([-150.0, -149.0, -148.0])
        lat, lon = _projected_to_latlon(y, x, 'EPSG:4326')
        assert lat.shape == (2, 3)
        assert lon.shape == (2, 3)
        np.testing.assert_array_equal(lat[:, 0], y)
        np.testing.assert_array_equal(lon[0, :], x)

    def test_projected_produces_geographic_range(self):
        """UTM coords should transform to reasonable lat/lon."""
        y = np.linspace(6_800_000, 6_801_000, 5)
        x = np.linspace(350_000, 351_000, 5)
        lat, lon = _projected_to_latlon(y, x, 'EPSG:32606')
        # Should be somewhere in Alaska
        assert lat.min() > 50
        assert lat.max() < 75
        assert lon.min() > -180
        assert lon.max() < -120


# ---------------------------------------------------------------------------
# get_hrrr_for_dataset (with mocked Herbie)
# ---------------------------------------------------------------------------


def _make_fake_hrrr_ds(projected_ds, t2m_kelvin=268.15):
    """Build a fake HRRR xr.Dataset matching what FastHerbie returns."""
    lat, lon = _projected_to_latlon(projected_ds.y.values, projected_ds.x.values, 'EPSG:32606')
    # Coarse HRRR-like grid covering the SAR area
    hrrr_lat_1d = np.linspace(lat.min() - 0.5, lat.max() + 0.5, 10)
    hrrr_lon_1d = np.linspace(lon.min() - 0.5, lon.max() + 0.5, 10)
    hrrr_lat2d, hrrr_lon2d = np.meshgrid(hrrr_lat_1d, hrrr_lon_1d, indexing='ij')
    return hrrr_lat2d, hrrr_lon2d, t2m_kelvin


class TestGetHrrrForDataset:
    def test_missing_crs_raises(self):
        ds = xr.Dataset(
            {'dem': (['y', 'x'], np.zeros((2, 2)))},
            coords={'time': pd.date_range('2025-01-01', periods=1), 'y': [0, 1], 'x': [0, 1]},
        )
        with pytest.raises(ValueError, match='CRS'):
            get_hrrr_for_dataset(ds)

    def test_missing_dem_raises(self, projected_ds):
        ds = projected_ds.drop_vars('dem')
        with pytest.raises(ValueError, match='dem'):
            get_hrrr_for_dataset(ds)

    def test_with_mocked_herbie(self, projected_ds, monkeypatch):
        """Full pipeline with mocked fetch_t2m_batch."""
        ny = projected_ds.sizes['y']
        nx = projected_ds.sizes['x']
        hrrr_lat2d, hrrr_lon2d, t2m_K_val = _make_fake_hrrr_ds(projected_ds)
        hrrr_ny, hrrr_nx = hrrr_lat2d.shape

        def mock_batch(cycle_datetimes, model='hrrrak', max_threads=10):
            nt = len(cycle_datetimes)
            times = pd.DatetimeIndex([pd.Timestamp(c) for c in cycle_datetimes])
            t2m_data = np.full((nt, hrrr_ny, hrrr_nx), t2m_K_val, dtype=np.float32)
            return xr.Dataset(
                {'t2m': (['time', 'y', 'x'], t2m_data)},
                coords={
                    'time': times,
                    'latitude': (['y', 'x'], hrrr_lat2d),
                    'longitude': (['y', 'x'], hrrr_lon2d),
                },
            )

        monkeypatch.setattr('sarvalanche.io.hrrr.fetch_t2m_batch', mock_batch)

        result = get_hrrr_for_dataset(projected_ds, model='hrrrak', max_threads=1)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'y', 'x')
        assert result.sizes['time'] == projected_ds.sizes['time']
        assert result.sizes['y'] == ny
        assert result.sizes['x'] == nx
        assert result.attrs['units'] == 'degC'
        # Should be close to -5°C (with lapse-rate adjustment)
        assert np.nanmean(result.values) < 0
        assert np.all(np.isfinite(result.values))

    def test_auto_detect_model(self, projected_ds, monkeypatch):
        """model=None should auto-detect to hrrrak for Alaska dataset."""
        hrrr_lat2d, hrrr_lon2d, _ = _make_fake_hrrr_ds(projected_ds)
        hrrr_ny, hrrr_nx = hrrr_lat2d.shape

        calls = []

        def mock_batch(cycle_datetimes, model='hrrrak', max_threads=10):
            calls.append(model)
            nt = len(cycle_datetimes)
            times = pd.DatetimeIndex([pd.Timestamp(c) for c in cycle_datetimes])
            return xr.Dataset(
                {'t2m': (['time', 'y', 'x'], np.full((nt, hrrr_ny, hrrr_nx), 270.0, dtype=np.float32))},
                coords={
                    'time': times,
                    'latitude': (['y', 'x'], hrrr_lat2d),
                    'longitude': (['y', 'x'], hrrr_lon2d),
                },
            )

        monkeypatch.setattr('sarvalanche.io.hrrr.fetch_t2m_batch', mock_batch)

        get_hrrr_for_dataset(projected_ds, model=None, max_threads=1)
        assert all(m == 'hrrrak' for m in calls)

    def test_failed_fetch_fills_nan(self, projected_ds, monkeypatch):
        """When fetch_t2m_batch returns None, all time steps should be NaN."""

        def mock_batch_fail(cycle_datetimes, model='hrrrak', max_threads=10):
            return None

        monkeypatch.setattr('sarvalanche.io.hrrr.fetch_t2m_batch', mock_batch_fail)

        result = get_hrrr_for_dataset(projected_ds, model='hrrrak', max_threads=1)
        assert np.all(np.isnan(result.values))

    def test_dedup_same_cycle(self, monkeypatch):
        """Two SAR timestamps mapping to the same HRRR cycle should share one fetch."""
        ny, nx = 5, 5
        y = np.linspace(6_760_000, 6_760_150, ny)
        x = np.linspace(355_000, 355_150, nx)
        # Both at 06:30 and 06:45 on same day → same 06:00 cycle
        times = pd.to_datetime(['2025-01-15 06:30', '2025-01-15 06:45'])
        ds = xr.Dataset(
            {
                'VV': (['time', 'y', 'x'], np.ones((2, ny, nx), dtype=np.float32)),
                'dem': (['y', 'x'], np.ones((ny, nx), dtype=np.float32) * 1000),
            },
            coords={'time': times, 'y': y, 'x': x},
        )
        ds = ds.rio.write_crs('EPSG:32606')

        lat, lon = _projected_to_latlon(y, x, 'EPSG:32606')
        hrrr_lat = np.linspace(lat.min() - 0.5, lat.max() + 0.5, 5)
        hrrr_lon = np.linspace(lon.min() - 0.5, lon.max() + 0.5, 5)
        hrrr_lat2d, hrrr_lon2d = np.meshgrid(hrrr_lat, hrrr_lon, indexing='ij')
        hrrr_ny, hrrr_nx = hrrr_lat2d.shape

        batch_calls = []

        def mock_batch(cycle_datetimes, model='hrrrak', max_threads=10):
            batch_calls.append(cycle_datetimes)
            nt = len(cycle_datetimes)
            times = pd.DatetimeIndex([pd.Timestamp(c) for c in cycle_datetimes])
            return xr.Dataset(
                {'t2m': (['time', 'y', 'x'], np.full((nt, hrrr_ny, hrrr_nx), 268.0, dtype=np.float32))},
                coords={
                    'time': times,
                    'latitude': (['y', 'x'], hrrr_lat2d),
                    'longitude': (['y', 'x'], hrrr_lon2d),
                },
            )

        monkeypatch.setattr('sarvalanche.io.hrrr.fetch_t2m_batch', mock_batch)

        result = get_hrrr_for_dataset(ds, model='hrrrak', max_threads=1)
        # Only 1 unique cycle should have been requested
        assert len(batch_calls) == 1
        assert len(batch_calls[0]) == 1
        # Both SAR timestamps should have the same values
        np.testing.assert_array_equal(result.isel(time=0).values, result.isel(time=1).values)


# ---------------------------------------------------------------------------
# Integration: real network — skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_fetch_t2m_batch_real():
    """Smoke-test FastHerbie batch download."""
    cycles = ['2025-01-15 06:00', '2025-01-16 18:00']
    ds = fetch_t2m_batch(cycles, model='hrrrak', max_threads=5)
    assert ds is not None
    assert 't2m' in ds
    assert ds.sizes['time'] == 2
    assert 'latitude' in ds.coords
    assert 'longitude' in ds.coords
    assert 200 < float(ds['t2m'].mean()) < 320
    ds.close()
