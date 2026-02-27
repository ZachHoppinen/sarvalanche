import numpy as np
import torch
import pytest

from sarvalanche.ml.SARTimeSeriesDataset import SARTimeSeriesDataset


def _make_scene(T=12, C=2, H=48, W=48):
    """Create a numpy array mimicking a SAR time series scene.

    Shape is (T, C, H, W) matching SARTimeSeriesDataset expectations.
    Must be wrapped in a list when passed to the constructor because
    bare numpy arrays are treated as iterables (split along axis 0).
    """
    return np.random.rand(T, C, H, W).astype(np.float32)


def test_dataset_length():
    """Dataset length should equal number of valid patches."""
    scene = _make_scene(T=5, C=2, H=32, W=32)
    ds = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2)
    # 32/16 = 2 patches per dim → 4 patches
    assert len(ds) == 4


def test_dataset_getitem_shapes():
    """Each item should have baseline (T, C, H, W) and target (C, H, W)."""
    scene = _make_scene(T=5, C=2, H=32, W=32)
    ds = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2)
    item = ds[0]
    assert 'baseline' in item
    assert 'target' in item
    assert item['baseline'].ndim == 4  # (T, C, H, W)
    assert item['target'].ndim == 3   # (C, H, W)
    assert item['baseline'].shape[1] == 2  # channels
    assert item['baseline'].shape[2] == 16  # patch H
    assert item['baseline'].shape[3] == 16  # patch W
    assert item['target'].shape == (2, 16, 16)


def test_dataset_baseline_length_range():
    """Baseline length should be between min_seq_len and min(max_seq_len, T-1)."""
    np.random.seed(42)
    scene = _make_scene(T=8, C=2, H=16, W=16)
    ds = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2, max_seq_len=5)
    lengths = set()
    for _ in range(100):
        item = ds[0]
        lengths.add(item['baseline'].shape[0])
    assert all(2 <= t <= 5 for t in lengths)


def test_dataset_nan_handling():
    """NaN values should be replaced with 0."""
    scene = _make_scene(T=5, C=2, H=16, W=16)
    scene[0, 0, :, :] = np.nan
    ds = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2)
    item = ds[0]
    assert not torch.any(torch.isnan(item['baseline']))
    assert not torch.any(torch.isnan(item['target']))


def test_dataset_multiple_scenes():
    """Should build patch index across multiple scenes."""
    scenes = [_make_scene(T=5, C=2, H=32, W=32), _make_scene(T=5, C=2, H=32, W=32)]
    ds = SARTimeSeriesDataset(scenes, patch_size=16, stride=16, min_seq_len=2)
    assert len(ds) == 8  # 4 patches per scene × 2 scenes


def test_dataset_skip_short_scene():
    """Scenes with T < min_seq_len + 1 should be skipped."""
    short_scene = _make_scene(T=2, C=2, H=16, W=16)
    ds = SARTimeSeriesDataset([short_scene], patch_size=16, stride=16, min_seq_len=2)
    assert len(ds) == 0


def test_dataset_overlapping_stride():
    """Smaller stride should produce more patches."""
    scene = _make_scene(T=5, C=2, H=48, W=48)
    ds_dense = SARTimeSeriesDataset([scene], patch_size=16, stride=8, min_seq_len=2)
    ds_sparse = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2)
    assert len(ds_dense) > len(ds_sparse)


def test_dataset_output_types():
    """Outputs should be float tensors."""
    scene = _make_scene(T=5, C=2, H=16, W=16)
    ds = SARTimeSeriesDataset([scene], patch_size=16, stride=16, min_seq_len=2)
    item = ds[0]
    assert item['baseline'].dtype == torch.float32
    assert item['target'].dtype == torch.float32
