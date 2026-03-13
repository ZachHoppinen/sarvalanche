"""Train the v2 single-pass debris detection CNN.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/train.py \
        --data-dir local/issw/dual_tau_output/zone/patches \
        --epochs 50 --pos-weight 10 --batch-size 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import DebrisDetector, DebrisDetectorSkip


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _grad_norms(model):
    """Compute gradient L2 norm per major submodule."""
    norms = {}
    for name, module in [('set_enc', model.set_encoder),
                         ('attn', model.attention),
                         ('static_enc', model.static_encoder),
                         ('fusion', model.fusion),
                         ('dec', model.decoder)]:
        total = 0.0
        count = 0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
                count += 1
        norms[name] = total ** 0.5 if count > 0 else 0.0
    return norms


def _format_grad_norms(norms):
    """Format gradient norms as a compact string."""
    parts = []
    for name, norm in norms.items():
        if norm > 0:
            parts.append(f'{name}={norm:.4f}')
    return '  '.join(parts) if parts else 'no grads'


def _weighted_bce(logits, targets, pos_weight):
    """BCE with pos_weight tensor for class imbalance."""
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight,
    )


def _dice_loss(logits, targets, smooth=1.0):
    """Soft Dice loss — directly optimizes overlap."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def train_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_grad_norms = {}

    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(sar_maps, static)
        loss = _weighted_bce(logits, targets, pos_weight) + _dice_loss(logits, targets)

        loss.backward()

        # Accumulate gradient norms
        batch_norms = _grad_norms(model)
        for k, v in batch_norms.items():
            epoch_grad_norms[k] = epoch_grad_norms.get(k, 0.0) + v

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    # Average gradient norms over batches
    if n_batches > 0:
        for k in epoch_grad_norms:
            epoch_grad_norms[k] /= n_batches

    return total_loss / max(n_batches, 1), epoch_grad_norms


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # IoU accumulators
    intersection = 0
    union = 0

    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        logits = model(sar_maps, static)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        total_loss += loss.item()
        n_batches += 1

        preds = (torch.sigmoid(logits) >= threshold).float()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()

    val_loss = total_loss / max(n_batches, 1)
    iou = intersection / max(union, 1)
    return val_loss, iou


def main():
    parser = argparse.ArgumentParser(description='Train v2 single-pass debris detector')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with v2 .npz patches')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detect if omitted)')
    parser.add_argument('--val-frac', type=float, default=0.15, help='Validation fraction')
    parser.add_argument('--pos-weight', type=float, default=0.0,
                        help='Positive class weight (0 = auto-compute from data)')
    parser.add_argument('--out', type=Path, default=None, help='Output weights path')
    parser.add_argument('--skip', action='store_true', help='Use DebrisDetectorSkip (skip connections)')
    args = parser.parse_args()

    device = _resolve_device(args.device)
    log.info('Using device: %s', device)

    # Dataset
    dataset = V2PatchDataset(args.data_dir, augment=True)
    if len(dataset) == 0:
        log.error('No patches found in %s', args.data_dir)
        return

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    val_ds.dataset.augment = False  # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=v2_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=v2_collate_fn, num_workers=0,
    )

    log.info('Train: %d, Val: %d', n_train, n_val)

    # Compute positive class weight from data
    if args.pos_weight > 0:
        pw = args.pos_weight
    else:
        # Auto-compute: count positive vs negative pixels across dataset
        n_pos_px = 0
        n_total_px = 0
        for i in range(len(dataset)):
            data = np.load(dataset.files[i], allow_pickle=True)
            if 'label_mask' in data:
                mask = data['label_mask']
                n_pos_px += mask.sum()
                n_total_px += mask.size
            else:
                label = int(data['label'])
                n_pos_px += label * 128 * 128
                n_total_px += 128 * 128
        n_neg_px = n_total_px - n_pos_px
        pw = float(n_neg_px / max(n_pos_px, 1))
        pw = min(pw, 50.0)  # cap to avoid extreme weights
    pos_weight = torch.tensor([pw], device=device)
    log.info('Positive class weight: %.1f', pw)

    # Model
    model_cls = DebrisDetectorSkip if args.skip else DebrisDetector
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info('Model parameters: %d', n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    out_path = args.out or (args.data_dir / 'v2_detector_best.pt')

    for epoch in range(args.epochs):
        train_loss, grad_norms = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_loss, iou = validate(model, val_loader, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info('  epoch %3d: train=%.4f  val=%.4f  IoU=%.4f  grads: %s',
                     epoch + 1, train_loss, val_loss, iou, _format_grad_norms(grad_norms))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            log.info('  Saved best model (val_loss=%.4f) at epoch %d', val_loss, epoch + 1)

    log.info('Training complete. Best val_loss=%.4f, saved to %s', best_val_loss, out_path)


if __name__ == '__main__':
    main()
