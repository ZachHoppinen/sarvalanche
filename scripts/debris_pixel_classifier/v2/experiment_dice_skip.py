"""A/B experiment: standard BCE vs BCE+Dice with skip-connection decoder.

Trains four configurations on the same train/val split and compares IoU:
  1. baseline     — DebrisDetector + weighted BCE  (current production)
  2. dice         — DebrisDetector + weighted BCE + Dice loss
  3. skip         — DebrisDetectorSkip (skip connections) + weighted BCE
  4. skip+dice    — DebrisDetectorSkip + weighted BCE + Dice loss

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/experiment_dice_skip.py \
        --data-dir local/issw/v2_patches --epochs 50 --batch-size 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from sarvalanche.ml.v2.channels import N_STATIC
from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import (
    ConvBlock,
    Decoder,
    DebrisDetector,
    SetEncoder,
    SpatialSetAttention,
    StaticEncoder,
    _DeconvBlock,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skip-connection model
# ---------------------------------------------------------------------------

class SetEncoderWithSkips(nn.Module):
    """SetEncoder that also returns intermediate feature maps for skip connections."""

    def __init__(self, in_ch: int = 2, feat_dim: int = 64):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 16, stride=2)   # → (16, 64, 64)
        self.block2 = ConvBlock(16, 32, stride=2)       # → (32, 32, 32)
        self.block3 = ConvBlock(32, feat_dim, stride=2)  # → (64, 16, 16)
        self.block4 = ConvBlock(feat_dim, feat_dim, stride=2)  # → (64, 8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)   # (B, 16, 64, 64)
        s2 = self.block2(s1)  # (B, 32, 32, 32)
        s3 = self.block3(s2)  # (B, 64, 16, 16)
        s4 = self.block4(s3)  # (B, 64, 8, 8)
        return s4, [s1, s2, s3]


class StaticEncoderWithSkips(nn.Module):
    """Static encoder returning intermediate features for skip connections."""

    def __init__(self, in_ch: int = N_STATIC):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 16, stride=4)  # → (16, 32, 32)
        self.block2 = ConvBlock(16, 24, stride=2)     # → (24, 16, 16)
        self.block3 = ConvBlock(24, 32, stride=2)     # → (32, 8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)   # (16, 32, 32)
        s2 = self.block2(s1)  # (24, 16, 16)
        s3 = self.block3(s2)  # (32, 8, 8)
        return s3, [s1, s2]


class _SkipDeconvBlock(nn.Module):
    """Upsample + concatenate skip features + refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch + skip_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        cat = torch.cat([up, skip], dim=1)
        return self.refine(cat)


class SkipDecoder(nn.Module):
    """Decoder with skip connections from encoder stages.

    Skip dimensions (SAR + static combined):
      stage 1 (16x16): SAR 64 + static 24 = 88
      stage 2 (32x32): SAR 32 + static 16 = 48
      stage 3 (64x64): SAR 16             = 16  (static has no 64x64 skip)
    """

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        # 8→16,  skip = 88 (SAR 64 + static 24)
        self.up1 = _SkipDeconvBlock(in_ch, 88, 32)
        # 16→32, skip = 48 (SAR 32 + static 16)
        self.up2 = _SkipDeconvBlock(32, 48, 16)
        # 32→64, skip = 16 (SAR 16 only)
        self.up3 = _SkipDeconvBlock(16, 16, 8)
        # 64→128, no skip
        self.up4 = _DeconvBlock(8, 4)
        self.head = nn.Conv2d(4, out_ch, 1)

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        """skips: [skip_16x16, skip_32x32, skip_64x64]"""
        x = self.up1(x, skips[0])   # 8→16
        x = self.up2(x, skips[1])   # 16→32
        x = self.up3(x, skips[2])   # 32→64
        x = self.up4(x)             # 64→128
        return self.head(x)


class DebrisDetectorSkip(nn.Module):
    """DebrisDetector variant with skip connections from encoders to decoder."""

    def __init__(self, feat_dim: int = 64, n_static: int = N_STATIC):
        super().__init__()
        self.set_encoder = SetEncoderWithSkips(in_ch=2, feat_dim=feat_dim)
        self.attention = SpatialSetAttention(feat_dim=feat_dim)
        self.static_encoder = StaticEncoderWithSkips(in_ch=n_static)
        self.fusion = nn.Conv2d(feat_dim + 32, feat_dim, 1)
        self.decoder = SkipDecoder(in_ch=feat_dim, out_ch=1)

        # Attention modules for pooling skip connections across variable SAR maps
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=16),   # 64x64 skips
            SpatialSetAttention(feat_dim=32),   # 32x32 skips
            SpatialSetAttention(feat_dim=64),   # 16x16 skips
        ])

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each SAR map, collecting skip features
        all_bottlenecks = []
        all_skips: list[list[torch.Tensor]] = [[], [], []]  # 3 skip levels

        for m in sar_maps:
            bottleneck, skips = self.set_encoder(m)
            all_bottlenecks.append(bottleneck)
            for i, s in enumerate(skips):
                all_skips[i].append(s)

        # Attention-pool bottleneck and each skip level across SAR maps
        sar_feat = self.attention(all_bottlenecks)  # (B, 64, 8, 8)
        sar_skips = [
            self.skip_attentions[i](all_skips[i])
            for i in range(3)
        ]
        # sar_skips: [(B,16,64,64), (B,32,32,32), (B,64,16,16)]

        # Static encoder with skips
        static_feat, static_skips = self.static_encoder(static)
        # static_skips: [(B,16,32,32), (B,24,16,16)]

        # Fuse bottleneck
        fused = torch.cat([sar_feat, static_feat], dim=1)
        fused = F.relu(self.fusion(fused))

        # Build combined skip connections for decoder
        # 16x16: SAR (64ch) + static (24ch) = 88ch
        skip_16 = torch.cat([sar_skips[2], static_skips[1]], dim=1)
        # 32x32: SAR (32ch) + static (16ch) = 48ch
        skip_32 = torch.cat([sar_skips[1], static_skips[0]], dim=1)
        # 64x64: SAR (16ch) only = 16ch
        skip_64 = sar_skips[0]

        return self.decoder(fused, [skip_16, skip_32, skip_64])


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def weighted_bce(logits, targets, pos_weight):
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def dice_loss(logits, targets, smooth=1.0):
    """Soft Dice loss — directly optimizes overlap."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def bce_dice_loss(logits, targets, pos_weight, dice_weight=1.0):
    """Combined weighted BCE + Dice loss."""
    bce = weighted_bce(logits, targets, pos_weight)
    dc = dice_loss(logits, targets)
    return bce + dice_weight * dc


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(sar_maps, static)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    """Compute val loss (unweighted BCE for fair comparison), IoU, and confidence stats."""
    model.eval()
    total_bce = 0.0
    n = 0
    intersection = 0
    union = 0
    # Track confidence on true-positive pixels
    tp_probs = []
    all_pos_probs = []

    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        logits = model(sar_maps, static)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        total_bce += bce.item()
        n += 1

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()

        # Confidence stats
        pos_mask = targets > 0.5
        if pos_mask.any():
            all_pos_probs.append(probs[pos_mask].cpu())
        tp_mask = (preds > 0.5) & (targets > 0.5)
        if tp_mask.any():
            tp_probs.append(probs[tp_mask].cpu())

    val_bce = total_bce / max(n, 1)
    iou = intersection / max(union, 1)

    # Mean confidence on positive-class pixels
    mean_pos_conf = torch.cat(all_pos_probs).mean().item() if all_pos_probs else 0.0
    mean_tp_conf = torch.cat(tp_probs).mean().item() if tp_probs else 0.0

    return val_bce, iou, mean_pos_conf, mean_tp_conf


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='A/B experiment: BCE vs BCE+Dice, with/without skip connections')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--pos-weight', type=float, default=0.0, help='0 = auto-compute')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info('Device: %s', device)

    # Dataset — single split shared across all runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = V2PatchDataset(args.data_dir, augment=True)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    val_ds.dataset.augment = False  # type: ignore[attr-defined]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=v2_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=v2_collate_fn, num_workers=0)

    log.info('Train: %d  Val: %d', n_train, n_val)

    # Positive class weight
    if args.pos_weight > 0:
        pw = args.pos_weight
    else:
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
        pw = min(pw, 50.0)
    log.info('pos_weight: %.1f', pw)
    pos_weight = torch.tensor([pw], device=device)

    # Define experiment configurations
    configs = {
        'baseline': {
            'model_cls': DebrisDetector,
            'loss_fn': lambda logits, targets: weighted_bce(logits, targets, pos_weight),
        },
        'dice': {
            'model_cls': DebrisDetector,
            'loss_fn': lambda logits, targets: bce_dice_loss(logits, targets, pos_weight),
        },
        'skip': {
            'model_cls': DebrisDetectorSkip,
            'loss_fn': lambda logits, targets: weighted_bce(logits, targets, pos_weight),
        },
        'skip+dice': {
            'model_cls': DebrisDetectorSkip,
            'loss_fn': lambda logits, targets: bce_dice_loss(logits, targets, pos_weight),
        },
    }

    results = {}

    for name, cfg in configs.items():
        log.info('=' * 60)
        log.info('Running: %s', name)
        log.info('=' * 60)

        # Reset seed for reproducible weight init
        torch.manual_seed(args.seed)

        model = cfg['model_cls']().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info('  Parameters: %d', n_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_iou = 0.0
        best_epoch = 0
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device, cfg['loss_fn'])
            val_bce, iou, mean_pos_conf, mean_tp_conf = validate(model, val_loader, device)
            scheduler.step()

            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info(
                    '  [%s] epoch %3d: train=%.4f  val_bce=%.4f  IoU=%.4f  '
                    'pos_conf=%.3f  tp_conf=%.3f',
                    name, epoch + 1, train_loss, val_bce, iou,
                    mean_pos_conf, mean_tp_conf,
                )

        # Final validation with best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)
        final_bce, final_iou, final_pos_conf, final_tp_conf = validate(model, val_loader, device)

        results[name] = {
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'final_val_bce': final_bce,
            'final_iou': final_iou,
            'mean_pos_confidence': final_pos_conf,
            'mean_tp_confidence': final_tp_conf,
            'n_params': n_params,
        }

        # Save best weights
        out_path = args.data_dir / f'v2_experiment_{name.replace("+", "_")}.pt'
        torch.save(best_state, out_path)
        log.info('  Saved %s → %s', name, out_path)

    # Summary
    log.info('')
    log.info('=' * 72)
    log.info('RESULTS SUMMARY')
    log.info('=' * 72)
    log.info('%-12s  %6s  %8s  %8s  %10s  %10s  %8s',
             'Config', 'Params', 'Val BCE', 'IoU', 'Pos Conf', 'TP Conf', 'BestEp')
    log.info('-' * 72)
    for name, r in results.items():
        log.info(
            '%-12s  %6d  %8.4f  %8.4f  %10.3f  %10.3f  %8d',
            name, r['n_params'], r['final_val_bce'], r['final_iou'],
            r['mean_pos_confidence'], r['mean_tp_confidence'], r['best_epoch'],
        )


if __name__ == '__main__':
    main()
