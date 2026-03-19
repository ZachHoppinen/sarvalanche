"""Train the v3 single-pair debris detector.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/train.py \
        --data-dir local/cnfaic/v3_experiment/patches \
        --epochs 50 --lr 1e-3 --base-ch 16
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sarvalanche.ml.v3.channels import N_INPUT
from sarvalanche.ml.v3.dataset import V3PairDataset
from sarvalanche.ml.v3.model import SinglePairDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _resolve_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def weighted_bce(logits, targets, pos_weight, sample_weights=None):
    per_pixel = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none',
    )
    if sample_weights is not None:
        w = sample_weights.view(-1, 1, 1, 1)
        return (per_pixel * w).mean()
    return per_pixel.mean()


def dice_loss(logits, targets, sample_weights=None, smooth=1.0):
    probs = torch.sigmoid(logits)
    if sample_weights is not None:
        B = logits.shape[0]
        losses = []
        for i in range(B):
            p = probs[i].flatten()
            t = targets[i].flatten()
            inter = (p * t).sum()
            d = 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
            losses.append(d * sample_weights[i])
        return sum(losses) / max(sample_weights.sum(), 1e-6)
    inter = (probs * targets).sum()
    return 1.0 - (2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch['x'].to(device)
        targets = batch['label'].to(device)
        weights = batch['confidence'].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = weighted_bce(logits, targets, pos_weight, weights) + dice_loss(logits, targets, weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n = 0
    intersection = union = 0

    for batch in loader:
        x = batch['x'].to(device)
        targets = batch['label'].to(device)

        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        total_loss += loss.item()
        n += 1

        preds = (torch.sigmoid(logits) >= threshold).float()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()

    return total_loss / max(n, 1), intersection / max(union, 1)


def main():
    parser = argparse.ArgumentParser(description="Train v3 single-pair debris detector")
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Dataset — train split only (val paths held out)
    from torch.utils.data import ConcatDataset, random_split

    datasets = []
    for d in args.data_dir:
        ds = V3PairDataset(d, augment=True, split='train')
        if len(ds) > 0:
            datasets.append(ds)
    if not datasets:
        log.error("No training patches found")
        return

    train_full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Random train/val split from non-path-overlap patches
    n_val = max(1, int(len(train_full) * args.val_frac))
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    log.info("Train: %d, Val: %d", n_train, n_val)

    # Auto-compute pos weight
    all_files = []
    for ds in datasets:
        all_files.extend(ds.files)
    n_pos = n_total = 0
    for f in all_files[:500]:  # sample for speed
        data = np.load(f, allow_pickle=True)
        if 'label_mask' in data:
            m = data['label_mask']
            n_pos += m.sum()
            n_total += m.size
    n_neg = n_total - n_pos
    pw = min(float(n_neg / max(n_pos, 1)), 50.0)
    pos_weight = torch.tensor([pw], device=device)
    log.info("Pos weight: %.1f", pw)

    # Model
    model = SinglePairDetector(in_ch=N_INPUT, base_ch=args.base_ch).to(device)
    if args.resume and args.resume.exists():
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        log.info("Resumed from %s", args.resume)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d params (base_ch=%d)", n_params, args.base_ch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    out_path = args.out or (args.data_dir[0] / 'v3_best.pt')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_loss, iou = validate(model, val_loader, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("epoch %3d: train=%.4f  val=%.4f  IoU=%.4f",
                     epoch + 1, train_loss, val_loss, iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            log.info("  Saved (val_loss=%.4f) at epoch %d", val_loss, epoch + 1)

    log.info("Done. Best val_loss=%.4f → %s", best_val_loss, out_path)


if __name__ == "__main__":
    main()
