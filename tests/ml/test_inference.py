import numpy as np
import torch
import xarray as xr
import pytest

from sarvalanche.ml.inference import prep_dataset_for_inference, predict_with_sweeping_fast
from sarvalanche.ml.SARTransformer import SARTransformer


# ── prep_dataset_for_inference ──────────────────────────────────────────────


def _make_sar_da(n_time=12, ny=10, nx=10):
    """Create a simple SAR DataArray with (time, y, x) dims."""
    return xr.DataArray(
        np.random.rand(n_time, ny, nx).astype(np.float32),
        dims=['time', 'y', 'x'],
        coords={
            'time': np.arange(n_time),
            'y': np.arange(ny),
            'x': np.arange(nx),
        },
    )


def test_prep_dataset_dims():
    """Output should have dims (time, polarization, y, x)."""
    vv = _make_sar_da(12)
    vh = _make_sar_da(12)
    result = prep_dataset_for_inference(vv, vh)
    assert result.dims == ('time', 'polarization', 'y', 'x')


def test_prep_dataset_truncates_to_10():
    """Only last 10 timesteps should be kept."""
    vv = _make_sar_da(15)
    vh = _make_sar_da(15)
    result = prep_dataset_for_inference(vv, vh)
    assert result.sizes['time'] == 10


def test_prep_dataset_short_series():
    """Shorter series should pass through without truncation."""
    vv = _make_sar_da(5)
    vh = _make_sar_da(5)
    result = prep_dataset_for_inference(vv, vh)
    assert result.sizes['time'] == 5


def test_prep_dataset_polarization_order():
    """Polarization coord should be ['VV', 'VH']."""
    vv = _make_sar_da(5)
    vh = _make_sar_da(5)
    result = prep_dataset_for_inference(vv, vh)
    assert list(result.coords['polarization'].values) == ['VV', 'VH']


# ── predict_with_sweeping_fast ──────────────────────────────────────────────


@pytest.fixture
def small_model():
    """A tiny transformer for testing inference."""
    model = SARTransformer(
        img_size=16, patch_size=8, in_chans=2,
        embed_dim=32, depth=1, num_heads=2,
        min_sigma=0.05, max_seq_len=10, dropout=0.0,
    )
    model.eval()
    return model


def test_predict_sweeping_output_shape(small_model):
    """Output shapes should match (C, H, W)."""
    baseline = np.random.rand(5, 2, 32, 32).astype(np.float32)
    mu, sigma = predict_with_sweeping_fast(
        small_model, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu',
    )
    assert mu.shape == (2, 32, 32)
    assert sigma.shape == (2, 32, 32)


def test_predict_sweeping_sigma_positive(small_model):
    """Sigma should be positive where predictions exist."""
    baseline = np.random.rand(3, 2, 32, 32).astype(np.float32)
    mu, sigma = predict_with_sweeping_fast(
        small_model, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu',
    )
    valid = np.isfinite(sigma)
    assert np.all(sigma[valid] > 0)


def test_predict_sweeping_nan_handling(small_model):
    """NaN pixels in input should produce NaN in output."""
    baseline = np.random.rand(3, 2, 32, 32).astype(np.float32)
    # Mark a block as NaN across all times and channels
    baseline[:, :, 0:5, 0:5] = np.nan
    mu, sigma = predict_with_sweeping_fast(
        small_model, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu',
    )
    # NaN region should remain NaN
    assert np.all(np.isnan(mu[:, 0:5, 0:5]))
    assert np.all(np.isnan(sigma[:, 0:5, 0:5]))


def test_predict_sweeping_xarray_input(small_model):
    """Should accept xr.DataArray and return xr.DataArray."""
    baseline = xr.DataArray(
        np.random.rand(3, 2, 32, 32).astype(np.float32),
        dims=['time', 'polarization', 'y', 'x'],
        coords={
            'polarization': ['VV', 'VH'],
            'y': np.arange(32),
            'x': np.arange(32),
        },
    )
    mu, sigma = predict_with_sweeping_fast(
        small_model, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu',
    )
    assert isinstance(mu, xr.DataArray)
    assert isinstance(sigma, xr.DataArray)
    assert mu.dims == ('polarization', 'y', 'x')


def test_predict_sweeping_min_valid_fraction(small_model):
    """Patches with insufficient valid pixels should be skipped."""
    baseline = np.random.rand(3, 2, 32, 32).astype(np.float32)
    # Make 90% of a 16x16 patch NaN
    baseline[:, :, 0:16, 0:14] = np.nan
    mu, sigma = predict_with_sweeping_fast(
        small_model, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu', min_valid_fraction=0.5,
    )
    # The mostly-NaN region should be NaN
    assert np.all(np.isnan(mu[:, 0:14, 0:14]))


def test_predict_sweeping_3d_input(small_model):
    """3D input (T, H, W) should be handled by expanding to (T, 1, H, W)."""
    # Need a model with in_chans=1 for this
    model_1ch = SARTransformer(
        img_size=16, patch_size=8, in_chans=1,
        embed_dim=32, depth=1, num_heads=2, dropout=0.0,
    )
    model_1ch.eval()
    baseline = np.random.rand(3, 32, 32).astype(np.float32)
    mu, sigma = predict_with_sweeping_fast(
        model_1ch, baseline, patch_size=16, stride=16,
        batch_size=4, device='cpu',
    )
    assert mu.shape == (1, 32, 32)
