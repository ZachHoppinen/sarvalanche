"""Permutation feature importance for the v2 CNN debris detector.

Shuffles each input feature across the validation set and measures the
degradation in IoU and increase in loss. Produces a summary table and
bar chart saved to the output directory.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/feature_importance.py \
        --data-dir local/issw/v2_patches \
        --weights local/issw/v2_patches/v2_detector_best.pt \
        --out-dir local/issw/v2_patches/feature_importance \
        --n-repeats 5
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sarvalanche.ml.v2.channels import STATIC_CHANNELS
from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import DebrisDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """Return (bce_loss, iou) on the given loader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
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

    return total_loss / max(n_batches, 1), intersection / max(union, 1)


class _PermutedStaticDataset(torch.utils.data.Dataset):
    """Wraps a dataset subset, permuting one static channel across samples."""

    def __init__(self, subset, channel_idx: int, rng: np.random.Generator):
        self.subset = subset
        self.channel_idx = channel_idx
        # Pre-compute a shuffled index mapping
        self.perm = rng.permutation(len(subset))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        # Replace the target channel with that from a random other sample
        donor = self.subset[int(self.perm[idx])]
        item['static'][self.channel_idx] = donor['static'][self.channel_idx]
        return item


class _PermutedSARDataset(torch.utils.data.Dataset):
    """Wraps a dataset subset, permuting SAR maps across samples."""

    def __init__(self, subset, rng: np.random.Generator, channel: str = 'both'):
        self.subset = subset
        self.channel = channel  # 'change', 'anf', or 'both'
        self.perm = rng.permutation(len(subset))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        donor = self.subset[int(self.perm[idx])]
        if self.channel == 'both':
            item['sar_maps'] = donor['sar_maps']
        elif self.channel == 'change':
            # Replace only the backscatter change channel (index 0)
            for i in range(len(item['sar_maps'])):
                if i < len(donor['sar_maps']):
                    item['sar_maps'][i][0] = donor['sar_maps'][i][0]
        elif self.channel == 'anf':
            # Replace only the ANF channel (index 1)
            for i in range(len(item['sar_maps'])):
                if i < len(donor['sar_maps']):
                    item['sar_maps'][i][1] = donor['sar_maps'][i][1]
        return item


def run_permutation_importance(model, val_subset, device, n_repeats=5, batch_size=8):
    """Compute permutation importance for all features.

    Returns dict mapping feature_name -> {'loss_increase': [...], 'iou_decrease': [...]}.
    """
    rng = np.random.default_rng(42)

    # Disable augmentation on the underlying dataset
    val_subset.dataset.augment = False

    # Baseline
    base_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             collate_fn=v2_collate_fn, num_workers=0)
    base_loss, base_iou = evaluate(model, base_loader, device)
    log.info('Baseline: loss=%.4f  IoU=%.4f', base_loss, base_iou)

    results = {}

    # --- Static channels ---
    for ch_idx, ch_name in enumerate(STATIC_CHANNELS):
        loss_increases = []
        iou_decreases = []
        for r in range(n_repeats):
            perm_ds = _PermutedStaticDataset(val_subset, ch_idx, rng)
            perm_loader = DataLoader(perm_ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=v2_collate_fn, num_workers=0)
            perm_loss, perm_iou = evaluate(model, perm_loader, device)
            loss_increases.append(perm_loss - base_loss)
            iou_decreases.append(base_iou - perm_iou)
        results[f'static: {ch_name}'] = {
            'loss_increase': loss_increases,
            'iou_decrease': iou_decreases,
        }
        log.info('  %s: dLoss=%.4f +/- %.4f  dIoU=%.4f +/- %.4f',
                 ch_name,
                 np.mean(loss_increases), np.std(loss_increases),
                 np.mean(iou_decreases), np.std(iou_decreases))

    # --- SAR features ---
    for sar_label, sar_ch in [('SAR: all', 'both'),
                               ('SAR: backscatter change', 'change'),
                               ('SAR: ANF', 'anf')]:
        loss_increases = []
        iou_decreases = []
        for r in range(n_repeats):
            perm_ds = _PermutedSARDataset(val_subset, rng, channel=sar_ch)
            perm_loader = DataLoader(perm_ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=v2_collate_fn, num_workers=0)
            perm_loss, perm_iou = evaluate(model, perm_loader, device)
            loss_increases.append(perm_loss - base_loss)
            iou_decreases.append(base_iou - perm_iou)
        results[sar_label] = {
            'loss_increase': loss_increases,
            'iou_decrease': iou_decreases,
        }
        log.info('  %s: dLoss=%.4f +/- %.4f  dIoU=%.4f +/- %.4f',
                 sar_label,
                 np.mean(loss_increases), np.std(loss_increases),
                 np.mean(iou_decreases), np.std(iou_decreases))

    return base_loss, base_iou, results


def plot_importance(results, base_loss, base_iou, out_dir: Path):
    """Create and save feature importance bar charts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    mean_iou_dec = [np.mean(results[n]['iou_decrease']) for n in names]
    std_iou_dec = [np.std(results[n]['iou_decrease']) for n in names]
    mean_loss_inc = [np.mean(results[n]['loss_increase']) for n in names]
    std_loss_inc = [np.std(results[n]['loss_increase']) for n in names]

    # Sort by mean IoU decrease (most important first)
    order = np.argsort(mean_iou_dec)[::-1]
    names_sorted = [names[i] for i in order]
    mean_iou_sorted = [mean_iou_dec[i] for i in order]
    std_iou_sorted = [std_iou_dec[i] for i in order]
    mean_loss_sorted = [mean_loss_inc[i] for i in order]
    std_loss_sorted = [std_loss_inc[i] for i in order]

    # Color: SAR features in blue, static in orange
    colors = ['#2196F3' if n.startswith('SAR') else '#FF9800' for n in names_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IoU decrease plot
    ax = axes[0]
    y_pos = np.arange(len(names_sorted))
    ax.barh(y_pos, mean_iou_sorted, xerr=std_iou_sorted, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('IoU Decrease (higher = more important)')
    ax.set_title(f'Permutation Feature Importance (IoU)\nBaseline IoU = {base_iou:.4f}')
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # Loss increase plot
    ax = axes[1]
    ax.barh(y_pos, mean_loss_sorted, xerr=std_loss_sorted, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Loss Increase (higher = more important)')
    ax.set_title(f'Permutation Feature Importance (Loss)\nBaseline Loss = {base_loss:.4f}')
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    log.info('Saved plot to %s', out_dir / 'feature_importance.png')
    plt.close(fig)

    # Save summary table as CSV
    import csv
    csv_path = out_dir / 'feature_importance.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'mean_iou_decrease', 'std_iou_decrease',
                         'mean_loss_increase', 'std_loss_increase'])
        for name in names_sorted:
            r = results[name]
            writer.writerow([
                name,
                f'{np.mean(r["iou_decrease"]):.6f}',
                f'{np.std(r["iou_decrease"]):.6f}',
                f'{np.mean(r["loss_increase"]):.6f}',
                f'{np.std(r["loss_increase"]):.6f}',
            ])
    log.info('Saved CSV to %s', csv_path)


def main():
    parser = argparse.ArgumentParser(description='Feature importance for v2 debris detector')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with v2 .npz patches')
    parser.add_argument('--weights', type=Path, required=True, help='Model weights (.pt)')
    parser.add_argument('--out-dir', type=Path, default=None, help='Output directory for plots/CSV')
    parser.add_argument('--n-repeats', type=int, default=5, help='Permutation repeats per feature')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Fraction of data for evaluation')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detect if omitted)')
    args = parser.parse_args()

    device = _resolve_device(args.device)
    log.info('Device: %s', device)

    # Load data
    dataset = V2PatchDataset(args.data_dir, augment=False)
    if len(dataset) == 0:
        log.error('No patches found in %s', args.data_dir)
        return

    # Use a subset for evaluation
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    _, val_subset = random_split(dataset, [n_train, n_val], generator=generator)
    log.info('Using %d patches for feature importance evaluation', n_val)

    # Load model
    model = DebrisDetector().to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    log.info('Loaded weights from %s', args.weights)

    # Run
    base_loss, base_iou, results = run_permutation_importance(
        model, val_subset, device,
        n_repeats=args.n_repeats, batch_size=args.batch_size,
    )

    # Output
    out_dir = args.out_dir or args.data_dir / 'feature_importance'
    plot_importance(results, base_loss, base_iou, out_dir)

    # Print summary
    print('\n' + '=' * 70)
    print(f'FEATURE IMPORTANCE SUMMARY  (baseline: loss={base_loss:.4f}, IoU={base_iou:.4f})')
    print('=' * 70)
    # Sort by IoU decrease
    sorted_names = sorted(results.keys(), key=lambda n: np.mean(results[n]['iou_decrease']), reverse=True)
    print(f'{"Feature":<30} {"dIoU (mean +/- std)":>22} {"dLoss (mean +/- std)":>22}')
    print('-' * 74)
    for name in sorted_names:
        r = results[name]
        diou = np.mean(r['iou_decrease'])
        diou_s = np.std(r['iou_decrease'])
        dloss = np.mean(r['loss_increase'])
        dloss_s = np.std(r['loss_increase'])
        print(f'{name:<30} {diou:>8.4f} +/- {diou_s:<8.4f} {dloss:>8.4f} +/- {dloss_s:<8.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
