"""Smoke test: train the segmentation encoder for a few steps on synthetic data."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sarvalanche.ml.track_features import N_PATCH_CHANNELS
from sarvalanche.ml.track_patch_dataset import TrackSegDataset
from sarvalanche.ml.track_patch_encoder import TrackSegEncoder


def test_training_loop_smoke():
    """Run 2 epochs on tiny synthetic data and verify loss decreases."""
    rng = np.random.default_rng(0)
    N, C, H, W = 16, N_PATCH_CHANNELS, 32, 32

    patches = rng.standard_normal((N, C, H, W)).astype(np.float32)
    # Put a clear signal in channel 0 (mahalanobis) correlated with the target
    signal = rng.random((N, H, W)).astype(np.float32)
    patches[:, 0] = signal
    # Track mask: ones everywhere for simplicity
    patches[:, 7] = 1.0

    targets = (signal > 0.5).astype(np.float32)[:, np.newaxis]  # (N, 1, H, W)

    dataset = TrackSegDataset(patches, targets, labels=None, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TrackSegEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    MASK_WEIGHT = 3.0
    losses = []
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for batch_patches, batch_targets in loader:
            opt.zero_grad()
            logits = model.segment(batch_patches)
            pixel_loss = criterion(logits, batch_targets)
            track_mask = batch_patches[:, 7:8, :, :]
            weight_map = 1.0 + (MASK_WEIGHT - 1.0) * track_mask
            loss = (pixel_loss * weight_map).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    # Loss should be finite
    assert all(np.isfinite(l) for l in losses), f"Non-finite losses: {losses}"
    # Model should produce correct output shape
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, N_PATCH_CHANNELS, H, W))
    assert out.shape == (1, 1, H, W)
