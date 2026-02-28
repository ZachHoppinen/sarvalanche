import numpy as np
import torch
import pytest

from sarvalanche.ml.track_patch_dataset import TrackPatchDataset, TrackSegDataset
from sarvalanche.ml.track_features import N_PATCH_CHANNELS


# ── TrackPatchDataset ───────────────────────────────────────────────────────


@pytest.fixture
def patch_data():
    N, C, H, W = 10, N_PATCH_CHANNELS, 64, 64
    patches = np.random.rand(N, C, H, W).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    return patches, labels


def test_patch_dataset_length(patch_data):
    ds = TrackPatchDataset(*patch_data)
    assert len(ds) == 10


def test_patch_dataset_getitem_types(patch_data):
    ds = TrackPatchDataset(*patch_data, augment=False)
    patch, label = ds[0]
    assert isinstance(patch, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert patch.shape == (N_PATCH_CHANNELS, 64, 64)
    assert label.ndim == 0


def test_patch_dataset_no_augment_preserves_data(patch_data):
    patches, labels = patch_data
    ds = TrackPatchDataset(patches, labels, augment=False)
    patch, label = ds[3]
    assert torch.allclose(patch, torch.from_numpy(patches[3]))
    assert label.item() == labels[3]


def test_patch_dataset_augment_flip():
    """Horizontal flip should negate easting channel (6)."""
    np.random.seed(42)
    patches = np.ones((1, N_PATCH_CHANNELS, 4, 4), dtype=np.float32)
    # Set easting channel to a known pattern: [1, 2, 3, 4] across columns
    patches[0, 6] = np.array([[1, 2, 3, 4]] * 4, dtype=np.float32)
    labels = np.array([1], dtype=np.float32)

    ds = TrackPatchDataset(patches, labels, augment=True, noise_std=0.0)

    # Sample many times to find a flip
    flipped = False
    for _ in range(50):
        p, _ = ds[0]
        east = p[6].numpy()
        # If flipped: first horizontally reversed [4,3,2,1] then negated [-4,-3,-2,-1]
        if east[0, 0] < 0:
            flipped = True
            assert east[0, 0] == pytest.approx(-4.0)
            assert east[0, -1] == pytest.approx(-1.0)
            break

    assert flipped, "Did not observe a flip in 50 attempts"


def test_patch_dataset_augment_noise():
    """Noise should only affect channels 0-3."""
    np.random.seed(0)
    patches = np.zeros((1, N_PATCH_CHANNELS, 8, 8), dtype=np.float32)
    labels = np.array([0], dtype=np.float32)
    ds = TrackPatchDataset(patches, labels, augment=True, noise_std=0.1)

    # Run several times; channels 5-7 should stay at 0 (when not flipped)
    for _ in range(20):
        p, _ = ds[0]
        # Channels 5 (northing) and 7 (track_mask) should be unchanged
        # (channel 6 might be negated by flip but magnitude stays 0)
        assert torch.allclose(p[7], torch.zeros(8, 8))


# ── TrackSegDataset ─────────────────────────────────────────────────────────


@pytest.fixture
def seg_data():
    N, C, H, W = 8, N_PATCH_CHANNELS, 64, 64
    patches = np.random.rand(N, C, H, W).astype(np.float32)
    targets = np.random.rand(N, 1, H, W).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    return patches, targets, labels


def test_seg_dataset_length(seg_data):
    ds = TrackSegDataset(seg_data[0], seg_data[1])
    assert len(ds) == 8


def test_seg_dataset_without_labels(seg_data):
    """Without labels, getitem should return (patch, target)."""
    ds = TrackSegDataset(seg_data[0], seg_data[1], labels=None, augment=False)
    result = ds[0]
    assert len(result) == 2
    patch, target = result
    assert patch.shape == (N_PATCH_CHANNELS, 64, 64)
    assert target.shape == (1, 64, 64)


def test_seg_dataset_with_labels(seg_data):
    """With labels, getitem should return (patch, target, label)."""
    ds = TrackSegDataset(*seg_data, augment=False)
    result = ds[0]
    assert len(result) == 3
    patch, target, label = result
    assert patch.shape == (N_PATCH_CHANNELS, 64, 64)
    assert target.shape == (1, 64, 64)
    assert label.ndim == 0


def test_seg_dataset_augment_flips_target(seg_data):
    """Augmentation flip should apply to both patch and target."""
    np.random.seed(42)
    patches = np.zeros((1, N_PATCH_CHANNELS, 4, 4), dtype=np.float32)
    targets = np.zeros((1, 1, 4, 4), dtype=np.float32)
    # Put a signal in left half
    targets[0, 0, :, 0:2] = 1.0
    ds = TrackSegDataset(patches, targets, augment=True, noise_std=0.0)

    saw_flip = False
    for _ in range(50):
        _, t = ds[0]
        if t[0, 0, -1].item() > 0.5:  # signal moved to right
            saw_flip = True
            break
    assert saw_flip, "Did not observe a target flip in 50 attempts"


def test_seg_dataset_no_augment_preserves_data(seg_data):
    patches, targets, labels = seg_data
    ds = TrackSegDataset(patches, targets, labels=labels, augment=False)
    p, t, l = ds[2]
    assert torch.allclose(p, torch.from_numpy(patches[2]))
    assert torch.allclose(t, torch.from_numpy(targets[2]))
    assert l.item() == labels[2]
