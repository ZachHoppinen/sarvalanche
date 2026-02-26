#!/usr/bin/env python
"""
Train the CNN segmentation encoder for debris-flow track classification.

Pure segmentation training: mask-weighted BCE loss against ``p_pixelwise``
soft targets (3x weight inside track polygon, 1x outside) using all ~85k
(track, date) patch–target pairs across all runs.

The trained seg encoder is used at inference to produce per-pixel debris
probabilities, which are aggregated into scalar features (seg_mean, seg_max,
etc.) and fed into the XGBoost track classifier.

Usage
-----
    conda run -n sarvalanche python scripts/train_track_encoder.py
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from sarvalanche.ml.track_classifier import build_all_seg_patches
from sarvalanche.ml.track_patch_dataset import TrackSegDataset
from sarvalanche.ml.track_patch_encoder import (
    CNN_ENCODER_DIR,
    CNN_SEG_ENCODER_PATH,
    TrackSegEncoder,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

RUNS_DIR = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')

EPOCHS       = 8
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MASK_WEIGHT  = 3.0   # extra weight for pixels inside the track polygon
CHECKPOINT_EVERY = 2  # save checkpoint every N epochs
TESTING      = False  # set to int to limit seg patches for quick testing


def _train_seg_model(
    patches: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    checkpoint_tag: str = "seg",
) -> TrackSegEncoder:
    """Train a segmentation encoder with mask-weighted BCE loss.

    Parameters
    ----------
    patches : np.ndarray of shape (N, C, H, W)
        Input patches.
    targets : np.ndarray of shape (N, 1, H, W)
        Soft segmentation targets (p_pixelwise).
    epochs : int
        Number of training epochs.
    checkpoint_tag : str
        Label for checkpoint filenames.
    """
    ckpt_dir = CNN_ENCODER_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TrackSegEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    seg_crit = nn.BCEWithLogitsLoss(reduction='none')

    loader = DataLoader(
        TrackSegDataset(patches, targets, labels=None, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    epoch_bar = tqdm(range(epochs), desc=f"training {checkpoint_tag}", unit="ep")
    for epoch in epoch_bar:
        epoch_loss = 0.0

        batch_bar = tqdm(loader, desc=f"  ep {epoch+1}/{epochs}",
                         unit="batch", leave=False)
        for batch_patches, batch_targets in batch_bar:
            opt.zero_grad()

            seg_logits = model.segment(batch_patches)             # (B, 1, H, W)
            pixel_loss = seg_crit(seg_logits, batch_targets)      # (B, 1, H, W)
            track_mask = batch_patches[:, 6:7, :, :]              # (B, 1, H, W)
            weight_map = 1.0 + (MASK_WEIGHT - 1.0) * track_mask
            loss = (pixel_loss * weight_map).mean()

            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()

        n = len(loader)
        epoch_bar.set_postfix(seg=f"{epoch_loss / n:.4f}")

        # Save checkpoint periodically
        if CHECKPOINT_EVERY and (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = ckpt_dir / f"{checkpoint_tag}_ep{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'seg_loss': epoch_loss / n,
            }, ckpt_path)
            log.info("  checkpoint saved → %s", ckpt_path)

    return model


def main() -> None:
    max_seg = TESTING if TESTING else None
    log.info("Building seg patches from ALL tracks in %s (max_tracks=%s)...", RUNS_DIR, max_seg)
    patches, targets = build_all_seg_patches(RUNS_DIR, max_tracks=max_seg)
    log.info("  patches=%s  targets=%s", patches.shape, targets.shape)

    if len(patches) == 0:
        log.error("No patches extracted, aborting.")
        return

    log.info("Training seg model (%d patches, %d epochs, batch_size=%d)...",
             len(patches), EPOCHS, BATCH_SIZE)
    model = _train_seg_model(patches, targets, EPOCHS)

    CNN_ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CNN_SEG_ENCODER_PATH)
    log.info("Model saved → %s", CNN_SEG_ENCODER_PATH)


if __name__ == '__main__':
    main()
