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
from torch.utils.data import DataLoader, Subset

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


def train_epoch(model, loader, optimizer, device, pos_weight, epoch=0, scaler=None):
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    n = 0
    use_amp = scaler is not None
    pbar = tqdm(loader, desc=f"Train ep{epoch+1}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        x = batch['x'].to(device)
        targets = batch['label'].to(device)
        weights = batch['confidence'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = weighted_bce(logits, targets, pos_weight, weights) + dice_loss(logits, targets, weights)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n += 1
        if n % 100 == 0:
            pbar.set_postfix(loss=f"{total_loss/n:.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    n = 0
    intersection = union = 0

    for batch in tqdm(loader, desc="Val", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test-mode", action="store_true",
                        help="Fast iteration mode: subsample to ~30 min total training")
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

    # Spatial split: group by patch position, assign positions to train/val
    # All pairs at a given (y0, x0) go together — prevents spatial leakage
    from collections import defaultdict
    pos_to_indices = defaultdict(list)
    for idx in range(len(train_full)):
        # Get the file path to extract position from filename
        if hasattr(train_full, 'datasets'):
            # ConcatDataset — find which sub-dataset and local index
            cumulative = 0
            for sub_ds in train_full.datasets:
                if idx < cumulative + len(sub_ds):
                    local_idx = idx - cumulative
                    fpath = sub_ds.files[local_idx]
                    break
                cumulative += len(sub_ds)
        else:
            fpath = train_full.files[idx]
        # Extract position: pos_0768_0128_v3_pair05.npz → pos_0768_0128
        fname = fpath.stem
        pos = fname.split('_v3_')[0]  # e.g. "pos_0768_0128" or "neg_0256_0384"
        pos_to_indices[pos].append(idx)

    # Randomly assign positions to train/val
    positions = list(pos_to_indices.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(positions)
    n_val_pos = max(1, int(len(positions) * args.val_frac))
    val_positions = set(positions[:n_val_pos])
    train_positions = set(positions[n_val_pos:])

    train_indices = [idx for pos in train_positions for idx in pos_to_indices[pos]]
    val_indices = [idx for pos in val_positions for idx in pos_to_indices[pos]]

    val_ds = Subset(train_full, val_indices)

    # Split train indices by label for per-epoch subsampling
    train_pos_indices = []
    train_neg_indices = []

    def _get_label(idx):
        if hasattr(train_full, 'datasets'):
            cumulative = 0
            for sub_ds in train_full.datasets:
                if idx < cumulative + len(sub_ds):
                    return sub_ds.labels[idx - cumulative]
                cumulative += len(sub_ds)
        return train_full.labels[idx]

    for idx in train_indices:
        if _get_label(idx) == 1:
            train_pos_indices.append(idx)
        else:
            train_neg_indices.append(idx)

    train_pos_indices = np.array(train_pos_indices)
    train_neg_indices = np.array(train_neg_indices)

    # Per-epoch: all positives + random subset of negatives (new each epoch)
    neg_sample_ratio = 1.0
    n_pos_per_epoch = len(train_pos_indices)

    if args.test_mode:
        # Fast mode: cap at ~28k samples/epoch for ~1 min/epoch
        max_samples = 28000
        n_pos_per_epoch = min(n_pos_per_epoch, max_samples // 4)
        log.info("TEST MODE: capping at %d samples/epoch", max_samples)

    n_neg_per_epoch = min(int(n_pos_per_epoch * neg_sample_ratio), len(train_neg_indices))
    samples_per_epoch = n_pos_per_epoch + n_neg_per_epoch

    log.info("Spatial split: %d positions (%d train, %d val)",
             len(positions), len(train_positions), len(val_positions))
    log.info("Train pool: %d pos + %d neg = %d total",
             len(train_pos_indices), len(train_neg_indices),
             len(train_pos_indices) + len(train_neg_indices))
    log.info("Per-epoch: %d pos + %d neg = %d samples (%.0f%% pos)",
             len(train_pos_indices), n_neg_per_epoch, samples_per_epoch,
             100 * len(train_pos_indices) / samples_per_epoch)

    # Cap validation to 5k samples for speed — same subset each epoch for consistency
    max_val = 5000
    if len(val_indices) > max_val:
        val_rng = np.random.default_rng(99)
        val_subset = val_rng.choice(val_indices, size=max_val, replace=False)
        val_ds = Subset(train_full, val_subset)
        log.info("Val capped to %d samples (from %d)", max_val, len(val_indices))
    else:
        val_ds = Subset(train_full, val_indices)

    pin = device.type == 'cuda'
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True, pin_memory=pin)

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

    # Mixed precision — works on CUDA, partial support on MPS
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        log.info("Using mixed precision (AMP)")

    best_val_loss = float('inf')
    out_path = args.out or (args.data_dir[0] / 'v3_best.pt')

    # Use a custom sampler that changes indices each epoch without recreating DataLoader
    from torch.utils.data import Sampler

    class EpochSubsampler(Sampler):
        """Samples all positives + random negatives, reshuffled each epoch."""
        def __init__(self, pos_indices, neg_indices, n_pos, n_neg):
            self.pos_indices = pos_indices
            self.neg_indices = neg_indices
            self.n_pos = n_pos
            self.n_neg = n_neg
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            rng = np.random.default_rng(self.epoch)
            if self.n_pos < len(self.pos_indices):
                pos = rng.choice(self.pos_indices, size=self.n_pos, replace=False)
            else:
                pos = self.pos_indices.copy()
            neg = rng.choice(self.neg_indices, size=self.n_neg, replace=False)
            all_idx = np.concatenate([pos, neg])
            rng.shuffle(all_idx)
            return iter(all_idx.tolist())

        def __len__(self):
            return self.n_pos + self.n_neg

    train_sampler = EpochSubsampler(train_pos_indices, train_neg_indices,
                                     n_pos_per_epoch, n_neg_per_epoch)

    # Single DataLoader with persistent workers — no restart between epochs
    # Sampler yields indices into train_full directly (not a Subset)
    train_loader = DataLoader(train_full, batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=10, persistent_workers=True, pin_memory=pin)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight, epoch, scaler)
        val_loss, iou = validate(model, val_loader, device)
        scheduler.step()

        log.info("epoch %3d: train=%.4f  val=%.4f  IoU=%.4f",
                 epoch + 1, train_loss, val_loss, iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            log.info("  Saved (val_loss=%.4f) at epoch %d", val_loss, epoch + 1)

    log.info("Done. Best val_loss=%.4f → %s", best_val_loss, out_path)


if __name__ == "__main__":
    main()
