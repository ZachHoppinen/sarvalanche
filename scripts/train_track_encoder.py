#!/usr/bin/env python
"""
Train the CNN segmentation encoder for debris-flow track classification.

Pure segmentation training: mask-weighted BCE loss against ``unmasked_p_target``
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from sarvalanche.ml.track_classifier import build_all_seg_patches
from sarvalanche.ml.track_features import TRACK_MASK_CHANNEL
from sarvalanche.ml.track_patch_dataset import TrackSegDataset
from sarvalanche.ml.track_patch_encoder import (
    CNN_ENCODER_DIR,
    CNN_SEG_ENCODER_PATH,
    TrackSegEncoder,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]

EPOCHS       = 24
BATCH_SIZE   = 64
LR           = 1e-3
LR_MIN       = 1e-5   # cosine annealing floor
WEIGHT_DECAY = 1e-4
MASK_WEIGHT  = 3.0    # extra weight for pixels inside the track polygon
CHECKPOINT_EVERY = 4  # save checkpoint every N epochs
TESTING      = False  # set to int to limit seg patches for quick testing
MAX_PRETRAIN = 10_000 # randomly subsample this many patches for pre-training (None = use all)
N_VIS        = 6      # number of example patches to visualize each epoch
VIS_DIR      = CNN_ENCODER_DIR / 'epoch_progress'


def _plot_epoch_examples(
    model: TrackSegEncoder,
    vis_patches: np.ndarray,
    vis_targets: np.ndarray,
    epoch: int,
    avg_loss: float,
    lr: float = 0.0,
) -> None:
    """Save a grid of example predictions for one epoch.

    Each example gets one row with 6 columns:
      Mahalanobis | Empirical p-value | Slope + Track | Target | CNN Prediction | Overlay
    """
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(vis_patches)

    model.eval()
    with torch.no_grad():
        logits = model.segment(torch.FloatTensor(vis_patches))
        probs = torch.sigmoid(logits).numpy()[:, 0]  # (N, H, W)
    model.train()

    fig, axes = plt.subplots(n, 6, figsize=(24, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f'Epoch {epoch + 1}  —  loss {avg_loss:.4f}  —  lr {lr:.2e}',
                 fontsize=14, y=1.0)

    for i in range(n):
        patch = vis_patches[i]        # (C, H, W)
        target = vis_targets[i, 0]    # (H, W)
        pred = probs[i]               # (H, W)

        # Col 0: Combined ML distance (z-score, normalized)
        ax = axes[i, 0]
        im = ax.imshow(patch[0], cmap='plasma', vmin=-1.2, vmax=1.2)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('ML Distance (z-score)')

        # Col 1: Backscatter change (dB, normalized)
        ax = axes[i, 1]
        im = ax.imshow(patch[1], cmap='RdYlGn_r', vmin=-2.0, vmax=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('Backscatter Δ (dB)')

        # Col 2: Slope + track outline (normalized by /0.6)
        ax = axes[i, 2]
        im = ax.imshow(patch[3], cmap='bone', vmin=0.4, vmax=1.3)
        ax.contour(patch[TRACK_MASK_CHANNEL], levels=[0.5], colors='red', linewidths=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('Slope (norm) + Track')

        # Col 3: Target
        ax = axes[i, 3]
        im = ax.imshow(target, cmap='hot', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('Target (unmasked_p_target)')

        # Col 4: CNN prediction
        ax = axes[i, 4]
        im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('CNN Prediction')

        # Col 5: Overlay — prediction contours on ML distance
        ax = axes[i, 5]
        ax.imshow(patch[0], cmap='plasma', vmin=-1.2, vmax=1.2)
        ax.contour(pred, levels=[0.3, 0.5, 0.7], colors=['cyan', 'yellow', 'red'],
                   linewidths=1.2)
        ax.contour(patch[TRACK_MASK_CHANNEL], levels=[0.5], colors='white', linewidths=0.8,
                   linestyles='dashed')
        if i == 0:
            ax.set_title('Pred contours on ML dist.')

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path = VIS_DIR / f'epoch_{epoch + 1:02d}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  epoch vis saved → %s", out_path)


def _train_seg_model_from_loader(
    dataset,
    epochs: int,
    vis_patches: np.ndarray | None = None,
    vis_targets: np.ndarray | None = None,
    checkpoint_tag: str = "seg",
) -> TrackSegEncoder:
    """Train a segmentation encoder with mask-weighted BCE loss.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset yielding (patch, target) tuples. Can be a ConcatDataset
        of disk-backed TrackSegDatasets.
    epochs : int
        Number of training epochs.
    vis_patches, vis_targets : np.ndarray or None
        Fixed examples for per-epoch visualization.
    checkpoint_tag : str
        Label for checkpoint filenames.
    """
    ckpt_dir = CNN_ENCODER_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TrackSegEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR_MIN)
    seg_crit = nn.BCEWithLogitsLoss(reduction='none')

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    epoch_bar = tqdm(range(epochs), desc=f"training {checkpoint_tag}", unit="ep")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        cur_lr = scheduler.get_last_lr()[0]

        batch_bar = tqdm(loader, desc=f"  ep {epoch+1}/{epochs}",
                         unit="batch", leave=False)
        for batch_patches, batch_targets in batch_bar:
            opt.zero_grad()

            seg_logits = model.segment(batch_patches)             # (B, 1, H, W)
            pixel_loss = seg_crit(seg_logits, batch_targets)      # (B, 1, H, W)
            track_mask = batch_patches[:, TRACK_MASK_CHANNEL:TRACK_MASK_CHANNEL+1, :, :]  # (B, 1, H, W)
            weight_map = 1.0 + (MASK_WEIGHT - 1.0) * track_mask
            loss = (pixel_loss * weight_map).mean()

            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()

        n = len(loader)
        scheduler.step()
        epoch_bar.set_postfix(seg=f"{epoch_loss / n:.4f}", lr=f"{cur_lr:.2e}")

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

        # Per-epoch visualization
        if vis_patches is not None:
            _plot_epoch_examples(model, vis_patches, vis_targets, epoch, epoch_loss / n, lr=cur_lr)

    return model


def main() -> None:
    from torch.utils.data import ConcatDataset

    max_seg = TESTING if TESTING else None

    # Use disk-backed binary files to avoid OOM with large patch counts.
    # Patches are streamed to raw .bin files during extraction, then
    # memory-mapped read-only for training (only accessed pages are resident).
    memmap_base = CNN_ENCODER_DIR / 'memmap_cache'
    memmap_base.mkdir(parents=True, exist_ok=True)

    from sarvalanche.ml.track_features import N_PATCH_CHANNELS

    datasets: list[TrackSegDataset] = []
    total_patches = 0
    all_patches_mms = []  # keep references alive for vis sampling
    SIZE = 64  # patch side length

    for i, runs_dir in enumerate(RUNS_DIRS):
        mm_dir = memmap_base / f'dir_{i}'
        patches_fp = mm_dir / 'patches.bin'
        targets_fp = mm_dir / 'targets.bin'

        # Reuse cached extraction if both .bin files exist and are non-empty
        if patches_fp.exists() and targets_fp.exists() and patches_fp.stat().st_size > 0:
            patch_bytes = N_PATCH_CHANNELS * SIZE * SIZE * 4
            target_bytes = 1 * SIZE * SIZE * 4
            n_p = patches_fp.stat().st_size // patch_bytes
            n_t = targets_fp.stat().st_size // target_bytes
            n = min(n_p, n_t)
            log.info("Loading cached extraction from %s (%d patches)", mm_dir, n)
            p = np.memmap(patches_fp, dtype=np.float32, mode='r',
                          shape=(n, N_PATCH_CHANNELS, SIZE, SIZE))
            t = np.memmap(targets_fp, dtype=np.float32, mode='r',
                          shape=(n, 1, SIZE, SIZE))
        else:
            log.info("Building seg patches from %s (max_tracks=%s, memmap=%s)...",
                     runs_dir, max_seg, mm_dir)
            p, t = build_all_seg_patches(runs_dir, max_tracks=max_seg, memmap_dir=mm_dir)

        log.info("  patches=%s  targets=%s", p.shape, t.shape)
        if len(p) > 0:
            datasets.append(TrackSegDataset(p, t, labels=None, augment=True))
            all_patches_mms.append((p, t))
            total_patches += len(p)

    log.info("  total patches=%d across %d dirs", total_patches, len(datasets))

    if total_patches == 0:
        log.error("No patches extracted, aborting.")
        return

    # Pick fixed examples for per-epoch visualization (spread across full dataset)
    vis_indices = np.linspace(0, total_patches - 1, N_VIS, dtype=int)
    vis_patches_list, vis_targets_list = [], []
    for vi in vis_indices:
        offset = 0
        for p, t in all_patches_mms:
            if vi < offset + len(p):
                vis_patches_list.append(np.array(p[vi - offset]))
                vis_targets_list.append(np.array(t[vi - offset]))
                break
            offset += len(p)
    vis_patches = np.stack(vis_patches_list)
    vis_targets = np.stack(vis_targets_list)
    log.info("  vis examples: indices %s", vis_indices.tolist())

    # Combine datasets without copying data
    combined_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Subsample to MAX_PRETRAIN patches if the full set is too large
    if MAX_PRETRAIN is not None and total_patches > MAX_PRETRAIN:
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(total_patches, size=MAX_PRETRAIN, replace=False)
        subset_idx.sort()  # sequential access is friendlier to memmap
        combined_ds = Subset(combined_ds, subset_idx.tolist())
        log.info("  subsampled %d → %d patches for pre-training", total_patches, MAX_PRETRAIN)
        total_patches = MAX_PRETRAIN

    log.info("Training seg model (%d patches, %d epochs, batch_size=%d)...",
             total_patches, EPOCHS, BATCH_SIZE)
    model = _train_seg_model_from_loader(combined_ds, EPOCHS,
                                          vis_patches=vis_patches, vis_targets=vis_targets)

    CNN_ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CNN_SEG_ENCODER_PATH)
    log.info("Model saved → %s", CNN_SEG_ENCODER_PATH)


if __name__ == '__main__':
    main()
