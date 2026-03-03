"""
Train the debris segmentation CNN.

Usage
-----
    conda run -n sarvalanche python scripts/cnn/train_seg.py

Trains a UNet++ segmentation model on context patches extracted from labeled
tracks. Uses weighted BCE loss with manual debris polygon labels blended with
unmasked_p_target as weak supervision.

Outputs
-------
  weights/seg_model_best.pt   — best validation loss checkpoint
  weights/seg_model_last.pt   — final epoch checkpoint
  weights/seg_training_log.csv — per-epoch metrics
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from sarvalanche.ml.losses import WeightedDebrisLoss
from sarvalanche.ml.debris_segmenter import DebrisSegmenter
from sarvalanche.ml.seg_training_data import build_seg_training_set
from sarvalanche.ml.weight_utils import find_weights, save_weights

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s – %(message)s')
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
SHAPES_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')
RUNS_DIRS   = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
WEIGHTS_DIR = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/weights')

PATCH_SIZE    = 64
BATCH_SIZE    = 32
N_EPOCHS      = 50
LR            = 1e-3
VAL_FRACTION  = 0.15
RANDOM_SEED   = 42

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device(
    'mps'  if torch.backends.mps.is_available()  else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)
log.info('Using device: %s', device)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DebrisDataset(Dataset):
    """Patch + target dataset with optional augmentation.

    Augmentations applied at training time:
      - Random horizontal flip
      - Random vertical flip
      - Random 90° rotation
    """

    def __init__(
        self,
        patches: np.ndarray,
        targets: np.ndarray,
        augment: bool = False,
    ):
        self.patches = torch.from_numpy(patches).float()
        self.targets = torch.from_numpy(targets).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch  = self.patches[idx]
        target = self.targets[idx]

        if self.augment:
            if torch.rand(1) > 0.5:
                patch  = torch.flip(patch,  dims=[-1])
                target = torch.flip(target, dims=[-1])
            if torch.rand(1) > 0.5:
                patch  = torch.flip(patch,  dims=[-2])
                target = torch.flip(target, dims=[-2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch  = torch.rot90(patch,  k=k, dims=[-2, -1])
                target = torch.rot90(target, k=k, dims=[-2, -1])

        return patch, target


# ── Build training data ───────────────────────────────────────────────────────

log.info('Building training data from %d runs dirs...', len(RUNS_DIRS))
with open(LABELS_PATH) as f:
    labels = json.load(f)

all_patches: list[np.ndarray] = []
all_targets: list[np.ndarray] = []
all_y:       list[pd.Series]  = []

for runs_dir in RUNS_DIRS:
    patches, targets, y = build_seg_training_set(
        labels, runs_dir, size=PATCH_SIZE, shapes_path=SHAPES_PATH
    )
    if len(patches) > 0:
        all_patches.append(patches)
        all_targets.append(targets)
        all_y.append(y)

patches = np.concatenate(all_patches, axis=0)
targets = np.concatenate(all_targets, axis=0)
y       = pd.concat(all_y)

log.info(
    'Total: %d patches, %.0f%% debris, patch shape %s',
    len(patches), 100 * y.mean(), patches.shape[1:],
)

# ── Train / val split ─────────────────────────────────────────────────────────

rng = torch.Generator().manual_seed(RANDOM_SEED)
n_val   = max(1, int(len(patches) * VAL_FRACTION))
n_train = len(patches) - n_val

full_ds    = DebrisDataset(patches, targets, augment=False)
train_ds_base, val_ds = random_split(full_ds, [n_train, n_val], generator=rng)

# Wrap train split with augmentation
train_indices = train_ds_base.indices
train_ds = DebrisDataset(
    patches[train_indices],
    targets[train_indices],
    augment=True,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

log.info('Train: %d  Val: %d  Batches/epoch: %d', n_train, n_val, len(train_loader))

# ── Model, optimiser, scheduler ───────────────────────────────────────────────

model     = DebrisSegmenter().to(device)
criterion = WeightedDebrisLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS, eta_min=1e-5)

log.info('Model parameters: %s', f'{sum(p.numel() for p in model.parameters()):,}')

# ── Training loop ─────────────────────────────────────────────────────────────

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
best_val_loss = float('inf')
log_rows: list[dict] = []

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.perf_counter()

    # Train
    model.train()
    train_losses: list[float] = []
    for patches_b, targets_b in train_loader:
        patches_b = patches_b.to(device)
        targets_b = targets_b.to(device)

        optimiser.zero_grad()
        logits = model(patches_b)
        loss   = criterion(logits, targets_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses:  list[float] = []
    val_ious:    list[float] = []

    with torch.no_grad():
        for patches_b, targets_b in val_loader:
            patches_b = patches_b.to(device)
            targets_b = targets_b.to(device)

            logits = model(patches_b)
            loss   = criterion(logits, targets_b)
            val_losses.append(loss.item())

            # IoU at threshold 0.5
            probs = torch.sigmoid(logits)
            pred  = (probs > 0.5).float()
            gt    = (targets_b > 0.5).float()
            intersection = (pred * gt).sum(dim=(-2, -1))
            union        = ((pred + gt) > 0).float().sum(dim=(-2, -1))
            iou = (intersection / union.clamp(min=1e-6)).mean().item()
            val_ious.append(iou)

    train_loss = float(np.mean(train_losses))
    val_loss   = float(np.mean(val_losses))
    val_iou    = float(np.mean(val_ious))
    lr_now     = scheduler.get_last_lr()[0]
    elapsed    = time.perf_counter() - t0

    scheduler.step()

    # Checkpoint
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_weights('debris_segmenter', 'best'))

    torch.save(model.state_dict(), save_weights('debris_segmenter', 'last'))

    log_rows.append({
        'epoch': epoch, 'train_loss': train_loss,
        'val_loss': val_loss, 'val_iou': val_iou,
        'lr': lr_now, 'best': is_best,
    })

    log.info(
        'Epoch %3d/%d  train=%.4f  val=%.4f  iou=%.3f  lr=%.2e  %s  %.1fs',
        epoch, N_EPOCHS, train_loss, val_loss, val_iou,
        lr_now, '★' if is_best else ' ', elapsed,
    )

# ── Save training log ─────────────────────────────────────────────────────────

log_path = WEIGHTS_DIR / 'seg_training_log.csv'
pd.DataFrame(log_rows).to_csv(log_path, index=False)
log.info('Training complete. Best val loss: %.4f', best_val_loss)
log.info('Saved weights to %s', WEIGHTS_DIR)
log.info('Saved training log to %s', log_path)