#!/usr/bin/env python
"""
Fine-tune the CNN segmentation encoder using labeled examples only.

Starts from the pretrained weights (trained on unmasked_p_target across all tracks)
and sharpens with supervised labels:

- **Positive targets**: Manually drawn debris polygons from ``debris_shapes.gpkg``
  are rasterized onto the patch grid. Where available, they are blended with
  ``unmasked_p_target`` via element-wise max.
- **Negative targets**: Tracks with label <= 1 (no debris) have their target
  zeroed inside the track polygon, teaching the model to suppress false positives.

Usage
-----
    conda run -n sarvalanche python scripts/finetune_track_encoder.py
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sarvalanche.ml.track_classifier import build_seg_training_set
from sarvalanche.ml.track_features import TRACK_MASK_CHANNEL
from sarvalanche.ml.track_patch_dataset import TrackSegDataset
from sarvalanche.ml.track_patch_encoder import (
    CNN_ENCODER_DIR,
    CNN_SEG_ENCODER_PATH,
    TrackSegEncoder,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

RUNS_DIR     = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')
LABELS_PATH  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
SHAPES_PATH  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')

# ── Fine-tuning hyperparameters ────────────────────────────────────────────────
EPOCHS            = 30
BATCH_SIZE        = 32       # smaller batches — fewer samples
LR                = 1e-4     # lower LR for fine-tuning
LR_MIN            = 1e-6
WEIGHT_DECAY      = 1e-4
MASK_WEIGHT       = 5.0      # stronger emphasis on track-interior pixels
NEG_MASK_WEIGHT   = 4.0      # extra weight for suppressing false positives in negatives
CHECKPOINT_EVERY  = 5
N_VIS             = 6
VIS_DIR           = CNN_ENCODER_DIR / 'finetune_progress'
FINETUNE_PATH     = CNN_ENCODER_DIR / 'track_seg_encoder_finetuned.pt'


def _plot_epoch_examples(
    model: TrackSegEncoder,
    vis_patches: np.ndarray,
    vis_targets: np.ndarray,
    vis_labels: np.ndarray,
    epoch: int,
    avg_loss: float,
    lr: float = 0.0,
) -> None:
    """Save a grid of example predictions for one epoch."""
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
    label_str = lambda lbl: 'debris' if lbl == 1 else 'no-debris'
    fig.suptitle(f'Fine-tune Epoch {epoch + 1}  —  loss {avg_loss:.4f}  —  lr {lr:.2e}',
                 fontsize=14, y=1.0)

    for i in range(n):
        patch = vis_patches[i]        # (C, H, W)
        target = vis_targets[i, 0]    # (H, W)
        pred = probs[i]               # (H, W)
        lbl = vis_labels[i]

        # Col 0: Combined ML distance (z-score, normalized)
        ax = axes[i, 0]
        im = ax.imshow(patch[0], cmap='plasma', vmin=-1.2, vmax=1.2)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('ML Distance (z-score)')
        ax.set_ylabel(label_str(lbl), fontsize=10, fontweight='bold')

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

        # Col 3: Target (labeled)
        ax = axes[i, 3]
        im = ax.imshow(target, cmap='hot', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        if i == 0:
            ax.set_title('Target (labeled)')

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
    out_path = VIS_DIR / f'finetune_epoch_{epoch + 1:02d}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  epoch vis saved → %s", out_path)


def main() -> None:
    # ── Load labels ────────────────────────────────────────────────────────────
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    # Filter to label <= 1 (negative) and label >= 2 (positive) — skip -1
    labels = {k: v for k, v in all_labels.items() if v['label'] >= 0}
    n_neg = sum(1 for v in labels.values() if v['label'] <= 1)
    n_pos = sum(1 for v in labels.values() if v['label'] >= 2)
    log.info("Labels: %d total (%d positive, %d negative)", len(labels), n_pos, n_neg)

    # ── Build labeled seg training set ─────────────────────────────────────────
    log.info("Building labeled seg patches (shapes_path=%s)...", SHAPES_PATH)
    patches, targets, y = build_seg_training_set(
        labels, RUNS_DIR, shapes_path=SHAPES_PATH,
    )
    log.info("  patches=%s  targets=%s  labels=%s", patches.shape, targets.shape, y.shape)

    if len(patches) == 0:
        log.error("No patches extracted, aborting.")
        return

    binary_labels = y.values  # (N,) int array: 0 or 1

    # ── Load pretrained weights ────────────────────────────────────────────────
    model = TrackSegEncoder()
    if CNN_SEG_ENCODER_PATH.exists():
        state = torch.load(CNN_SEG_ENCODER_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        log.info("Loaded pretrained weights from %s", CNN_SEG_ENCODER_PATH)
    else:
        log.warning("No pretrained weights found at %s — training from scratch",
                     CNN_SEG_ENCODER_PATH)

    # ── Pick visualization examples ────────────────────────────────────────────
    # Choose a mix of positive and negative examples
    pos_idx = np.where(binary_labels == 1)[0]
    neg_idx = np.where(binary_labels == 0)[0]
    n_vis_pos = min(N_VIS // 2, len(pos_idx))
    n_vis_neg = min(N_VIS - n_vis_pos, len(neg_idx))
    vis_idx = np.concatenate([
        np.random.default_rng(42).choice(pos_idx, n_vis_pos, replace=False),
        np.random.default_rng(42).choice(neg_idx, n_vis_neg, replace=False),
    ])
    vis_patches = patches[vis_idx]
    vis_targets = targets[vis_idx]
    vis_labels = binary_labels[vis_idx]
    log.info("  vis examples: %d pos, %d neg", n_vis_pos, n_vis_neg)

    # ── Training setup ─────────────────────────────────────────────────────────
    ckpt_dir = CNN_ENCODER_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    seg_crit = nn.BCEWithLogitsLoss(reduction='none')

    loader = DataLoader(
        TrackSegDataset(patches, targets, labels=binary_labels, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # ── Fine-tuning loop ───────────────────────────────────────────────────────
    model.train()
    epoch_bar = tqdm(range(EPOCHS), desc="fine-tuning", unit="ep")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        cur_lr = scheduler.get_last_lr()[0]

        batch_bar = tqdm(loader, desc=f"  ep {epoch+1}/{EPOCHS}",
                         unit="batch", leave=False)
        for batch_patches, batch_targets, batch_labels in batch_bar:
            opt.zero_grad()

            seg_logits = model.segment(batch_patches)             # (B, 1, H, W)
            pixel_loss = seg_crit(seg_logits, batch_targets)      # (B, 1, H, W)

            # Mask-weighted loss: heavier inside track polygon
            track_mask = batch_patches[:, TRACK_MASK_CHANNEL:TRACK_MASK_CHANNEL+1, :, :]  # (B, 1, H, W)
            weight_map = 1.0 + (MASK_WEIGHT - 1.0) * track_mask

            # Extra weight for negative samples inside their track polygon
            # to strongly suppress false positives
            is_neg = (batch_labels < 0.5).float()[:, None, None, None]  # (B, 1, 1, 1)
            weight_map = weight_map + NEG_MASK_WEIGHT * is_neg * track_mask

            loss = (pixel_loss * weight_map).mean()

            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()

        n_batches = len(loader)
        scheduler.step()
        epoch_bar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}", lr=f"{cur_lr:.2e}")

        # Checkpoint
        if CHECKPOINT_EVERY and (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = ckpt_dir / f"finetune_ep{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss / n_batches,
            }, ckpt_path)
            log.info("  checkpoint saved → %s", ckpt_path)

        # Per-epoch visualization
        _plot_epoch_examples(model, vis_patches, vis_targets, vis_labels,
                             epoch, epoch_loss / n_batches, lr=cur_lr)

    # ── Save final model ───────────────────────────────────────────────────────
    CNN_ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FINETUNE_PATH)
    log.info("Fine-tuned model saved → %s", FINETUNE_PATH)

    # Also overwrite the main seg encoder path so downstream uses the finetuned version
    torch.save(model.state_dict(), CNN_SEG_ENCODER_PATH)
    log.info("Also saved to main path → %s", CNN_SEG_ENCODER_PATH)


if __name__ == '__main__':
    main()
