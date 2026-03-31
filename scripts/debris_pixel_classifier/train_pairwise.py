"""Train pairwise debris detector on multi-zone SNFAC data.

Loads all zone-seasons lazily from NetCDFs listed in a YAML config.
Patches are sliced from disk on the fly — no full datasets held in memory.

Requires NetCDFs prepared via prepare_netcdf.py (adds derived static channels).

Usage:
    # Prepare netcdfs first:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/prepare_netcdf.py \
        --config scripts/debris_pixel_classifier/train_config_combined.yaml

    # Then train:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/train_pairwise.py \
        --config scripts/debris_pixel_classifier/train_config_combined.yaml \
        --epochs 50 \
        --out src/sarvalanche/ml/weights/pairwise_debris_detector/best.pt

    # Quick smoke test:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/train_pairwise.py \
        --config scripts/debris_pixel_classifier/train_config_alaska.yaml \
        --epochs 2 --test-mode
"""

import argparse
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset
from tqdm import tqdm

from sarvalanche.ml.pairwise_debris_classifier.channels import (
    N_INPUT, N_SAR, N_STATIC, SAR_CHANNELS, STATIC_CHANNELS,
)
from sarvalanche.ml.pairwise_debris_classifier.dataset import (
    PairwiseDebrisDataset, build_lazy_dataset,
)
from sarvalanche.ml.pairwise_debris_classifier.model import SinglePairDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ── Loss functions ───────────────────────────────────────────────────

def weighted_bce(logits, targets, pos_weight):
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight,
    )


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    return 1.0 - (2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)


# ── Training / validation loops ──────────────────────────────────────

def train_epoch(model, loader, optimizer, device, pos_weight, epoch=0):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Train ep{epoch+1}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        x = batch["x"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = weighted_bce(logits, targets, pos_weight) + dice_loss(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
        if n % 50 == 0:
            pbar.set_postfix(loss=f"{total_loss/n:.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, thresholds=(0.2, 0.3, 0.5)):
    model.eval()
    total_loss = 0.0
    n = 0
    tp = {t: 0 for t in thresholds}
    pred_pos = {t: 0 for t in thresholds}
    actual_pos = 0
    total_px = 0
    for batch in tqdm(loader, desc="Val", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        x = batch["x"].to(device)
        targets = batch["label"].to(device)
        logits = model(x)
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, targets)
                + dice_loss(logits, targets))
        total_loss += loss.item()
        n += 1
        probs = torch.sigmoid(logits)
        actual_pos += targets.sum().item()
        total_px += targets.numel()
        for t in thresholds:
            preds = (probs >= t).float()
            tp[t] += (preds * targets).sum().item()
            pred_pos[t] += preds.sum().item()

    ious = {}
    for t in thresholds:
        union = tp[t] + (actual_pos - tp[t]) + (pred_pos[t] - tp[t])
        ious[t] = tp[t] / max(union, 1)

    # Log precision, recall, F1 at each threshold
    parts = []
    for t in sorted(thresholds):
        p = tp[t] / max(pred_pos[t], 1)
        r = tp[t] / max(actual_pos, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        parts.append(f"@{t}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    pred_summary = f"pred>0.5={int(pred_pos[0.5])/1000:.0f}K/{total_px/1e6:.0f}M"
    log.info("  val: %s  %s", "  ".join(parts), pred_summary)

    return total_loss / max(n, 1), ious


# ── Curriculum-aware sampler ─────────────────────────────────────────

class CurriculumSampler(Sampler):
    """Samples curriculum-filtered train indices, excluding val.

    Passes total_epochs and auto_label_frac through to each dataset's
    get_valid_indices so curriculum and autolabel phasing work correctly.
    """

    def __init__(self, datasets, concat_offsets, val_indices_set,
                 total_epochs=100, auto_label_frac=0.6):
        self.datasets = datasets
        self.concat_offsets = concat_offsets
        self.val_indices_set = val_indices_set
        self.total_epochs = total_epochs
        self.auto_label_frac = auto_label_frac
        self.epoch = 0
        self._cached_epoch = -1
        self._cached_pos = []
        self._cached_neg = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_indices(self):
        if self._cached_epoch == self.epoch:
            return self._cached_pos, self._cached_neg
        pos_indices = []
        neg_indices = []
        for ds, offset in zip(self.datasets, self.concat_offsets):
            valid = ds.get_valid_indices(
                self.epoch, total_epochs=self.total_epochs,
                auto_label_frac=self.auto_label_frac)
            for local_idx in valid:
                global_idx = offset + local_idx
                if global_idx in self.val_indices_set:
                    continue
                if ds.labels[local_idx]:
                    pos_indices.append(global_idx)
                else:
                    neg_indices.append(global_idx)
        self._cached_epoch = self.epoch
        self._cached_pos = pos_indices
        self._cached_neg = neg_indices
        return pos_indices, neg_indices

    def __iter__(self):
        pos, neg = self._get_indices()
        rng = np.random.default_rng(self.epoch)
        n_neg = min(len(pos), len(neg))
        if len(neg) > n_neg:
            neg = rng.choice(neg, size=n_neg, replace=False).tolist()
        all_idx = pos + neg
        rng.shuffle(all_idx)
        return iter(all_idx)

    def __len__(self):
        pos, neg = self._get_indices()
        return len(pos) + min(len(pos), len(neg))


# ── Data loading ─────────────────────────────────────────────────────

def load_zone_datasets(zone_cfg, max_span_days=60, stride=64, month_range=(10, 4),
                       sar_only=False, post_context=False):
    """Build train (augmented) and val (clean) lazy datasets for one zone.

    Builds metadata once via build_lazy_dataset, then creates a second
    dataset sharing the same metadata but with augment=False for val.
    """
    name = zone_cfg['name']
    nc_path = Path(zone_cfg['nc'])
    log.info("Building dataset for zone %s", name)

    date_polygon_pairs = []

    for label_cfg in zone_cfg.get('labels', []):
        date_str = label_cfg['date']
        polygons_path = label_cfg['polygons']
        gdf = gpd.read_file(polygons_path)
        gt_dir = Path(label_cfg['geotiff_dir']) if label_cfg.get('geotiff_dir') else None
        # Store source filename for pair info parsing
        gdf.attrs['source'] = str(polygons_path)
        date_polygon_pairs.append((date_str, gdf, gt_dir, False))  # is_auto=False
        log.info("  Manual label: %s (%d polys)", date_str, len(gdf))

    autolabel_path = zone_cfg.get('autolabels')
    if autolabel_path and Path(autolabel_path).exists():
        auto_gdf = gpd.read_file(autolabel_path)
        if 't_end' in auto_gdf.columns:
            for date_str, group in auto_gdf.groupby('t_end'):
                date_polygon_pairs.append((date_str, group, None, True))  # is_auto=True
            log.info("  Autolabels: %d polys across %d dates",
                     len(auto_gdf), auto_gdf['t_end'].nunique())

    if not date_polygon_pairs:
        log.warning("  No labels for zone %s, skipping", name)
        return None, None

    t0 = time.time()
    # Build once with augmentation for training
    train_ds = build_lazy_dataset(
        nc_path, date_polygon_pairs,
        max_span_days=max_span_days, stride=stride, augment=True,
        month_range=month_range, sar_only=sar_only, post_context=post_context,
    )
    # Val dataset shares same metadata, just no augmentation
    val_ds = PairwiseDebrisDataset(
        nc_path, train_ds.pair_metas, train_ds.positions,
        train_ds.date_configs, anf_track_indices=train_ds.anf_track_indices,
        augment=False, sar_only=sar_only, post_context=post_context,
    )
    log.info("  Zone %s: %d samples (%.0fs)", name, len(train_ds), time.time() - t0)
    return train_ds, val_ds


# ── Main ─────────────────────────────────────────────────────────────

def _resolve_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser(description="Train pairwise debris detector")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("src/sarvalanche/ml/weights/pairwise_debris_detector/best.pt"))
    parser.add_argument("--auto-label-frac", type=float, default=0.6,
                        help="Fraction of epochs using autolabels (default 0.6, last 40%% human only)")
    parser.add_argument("--month-range", type=int, nargs=2, default=[10, 4], metavar=('START', 'END'),
                        help="Only include pairs ending in these months (wraps around year, default: 10 4 = Oct-Apr)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint (loads model weights, optimizer, epoch)")
    parser.add_argument("--pretrain", type=Path, default=None,
                        help="Load pretrained weights but start from epoch 0 with fresh optimizer")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--sar-only", action="store_true",
                        help="Use only SAR channels (4ch), drop all static channels")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout2d rate after encoder blocks (default 0.0 = off)")
    parser.add_argument("--post-context", action="store_true",
                        help="Add post-event VV/VH channels (6 SAR + 5 static = 11ch)")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build lazy datasets for all zones (train + val versions)
    t_load = time.time()
    n_zones = len(config['zones'])
    train_datasets = []
    val_datasets = []
    zone_names = []
    all_dates = []
    for zi, zone_cfg in enumerate(config['zones']):
        log.info("=== Zone %d/%d: %s ===", zi + 1, n_zones, zone_cfg['name'])
        train_ds, val_ds = load_zone_datasets(
            zone_cfg, max_span_days=args.max_span_days, stride=args.stride,
            month_range=tuple(args.month_range), sar_only=args.sar_only,
            post_context=args.post_context)
        if train_ds is not None:
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            zone_names.append(zone_cfg['name'])
            for label_cfg in zone_cfg.get('labels', []):
                all_dates.append(label_cfg['date'])

    if not train_datasets:
        log.error("No datasets loaded, exiting")
        return

    log.info("Loaded %d/%d zones, %d total samples in %.0fs",
             len(train_datasets), n_zones,
             sum(len(d) for d in train_datasets), time.time() - t_load)
    for name, ds in zip(zone_names, train_datasets):
        n_pos = sum(ds.labels)
        log.info("  %s: %d samples (%d pos, %d neg)",
                 name, len(ds), n_pos, len(ds) - n_pos)

    # Concat offsets for global indexing
    concat_offsets = []
    offset = 0
    for ds in train_datasets:
        concat_offsets.append(offset)
        offset += len(ds)
    combined_train = ConcatDataset(train_datasets)

    val_concat_offsets = []
    offset = 0
    for ds in val_datasets:
        val_concat_offsets.append(offset)
        offset += len(ds)
    combined_val = ConcatDataset(val_datasets)

    # Spatial val split — human labels only
    # Autolabels always go to training. Val split only applies to human-labeled positions.
    human_pos_keys = []
    human_pos_to_global = {}
    for di, (ds, off) in enumerate(zip(train_datasets, concat_offsets)):
        for local_idx, pos in enumerate(ds.positions):
            _, has_debris, y0, x0, _, is_auto = pos
            if is_auto:
                continue  # autolabels skip the val split entirely
            key = f"{zone_names[di]}_{y0:04d}_{x0:04d}"
            if key not in human_pos_to_global:
                human_pos_to_global[key] = []
                human_pos_keys.append(key)
            human_pos_to_global[key].append(off + local_idx)

    rng = np.random.default_rng(42)
    rng.shuffle(human_pos_keys)
    n_val = max(1, int(len(human_pos_keys) * args.val_frac))
    val_keys = set(human_pos_keys[:n_val])

    # Val indices = human-labeled samples at val spatial locations
    val_indices = []
    val_indices_from_train = set()
    for key in val_keys:
        indices = human_pos_to_global[key]
        val_indices.extend(indices)
        val_indices_from_train.update(indices)

    log.info("Val split: %d human positions (%d train, %d val), autolabels all in train",
             len(human_pos_keys), len(human_pos_keys) - n_val, n_val)

    # Cap val to maintain reasonable eval time
    max_val = 15000
    if len(val_indices) > max_val:
        val_indices = rng.choice(val_indices, max_val, replace=False).tolist()

    # Val loader uses the unaugmented datasets
    val_ds = Subset(combined_val, val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True)

    log.info("Spatial split: %d train positions, %d val positions (%d val samples)",
             len(human_pos_keys) - n_val, n_val, len(val_indices))

    # Pos weight from pixel ratio (use unaugmented val dataset for clean sampling)
    log.info("Sampling pixel ratio from positive patches...")
    n_pos_px = n_total_px = 0
    all_pos_indices = []
    for ds, off in zip(val_datasets, val_concat_offsets):
        for local_idx, label in enumerate(ds.labels):
            if label:
                all_pos_indices.append(off + local_idx)
    sample_size = min(200, len(all_pos_indices))
    if sample_size > 0:
        sample_idx = rng.choice(all_pos_indices, size=sample_size, replace=False)
        for si in sample_idx:
            batch = combined_val[si]
            mask = batch['label'].numpy()
            n_pos_px += mask.sum()
            n_total_px += mask.size
        n_neg_px = n_total_px - n_pos_px
        pw = min(50.0, float(n_neg_px / max(n_pos_px, 1)))
    else:
        pw = 10.0
    pos_weight = torch.tensor([pw], device=device)
    log.info("Pos weight: %.1f", pw)

    # Model
    n_post = 2 if args.post_context else 0
    if args.sar_only:
        in_ch = N_SAR + n_post
    else:
        in_ch = N_INPUT + n_post
    model = SinglePairDetector(in_ch=in_ch, base_ch=args.base_ch, dropout=args.dropout).to(device)
    log.info("Model: %d params (in_ch=%d, base_ch=%d%s%s%s)",
             sum(p.numel() for p in model.parameters()), in_ch, args.base_ch,
             ", SAR-only" if args.sar_only else "",
             ", +post_vv/vh" if args.post_context else "",
             f", dropout={args.dropout}" if args.dropout > 0 else "")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Warm restarts aligned with curriculum transitions:
    # T_0=10 → restart at epoch 10 (curriculum L0→L1)
    # T_mult=2 → next restart at epoch 30 (near L1→L2 at 20), then 70
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01,
    )

    best_val_loss = float("inf")
    start_epoch = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    latest_path = args.out.parent / (args.out.stem + "_latest" + args.out.suffix)

    # Load pretrained weights (fresh start at epoch 0)
    if args.pretrain and args.pretrain.exists():
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            log.info("Pretrained from %s (epoch %s, val_loss=%.4f) — starting fresh at epoch 0",
                     args.pretrain, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
        else:
            model.load_state_dict(ckpt)
            log.info("Pretrained weights from %s — starting fresh at epoch 0", args.pretrain)

    # Resume from checkpoint (restores epoch, optimizer, scheduler)
    elif args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            log.info("Resumed from %s (epoch %d, val_loss=%.4f)",
                     args.resume, start_epoch, best_val_loss)
        else:
            model.load_state_dict(ckpt)
            log.info("Resumed weights from %s (no epoch info)", args.resume)
        for _ in range(start_epoch):
            scheduler.step()

    ckpt_meta = {
        "model_config": {
            "architecture": "SinglePairDetector",
            "in_ch": in_ch,
            "base_ch": args.base_ch,
            "n_sar": N_SAR,
            "n_static": 0 if args.sar_only else N_STATIC,
            "sar_channels": list(SAR_CHANNELS),
            "static_channels": [] if args.sar_only else list(STATIC_CHANNELS),
            "sar_only": args.sar_only,
            "dropout": args.dropout,
            "post_context": args.post_context,
        },
        "zones": zone_names,
        "seasons": all_dates,
        "training_args": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "stride": args.stride,
            "val_frac": args.val_frac,
            "epochs": args.epochs,
            "max_span_days": args.max_span_days,
            "pos_weight": pw,
            "n_zones": len(train_datasets),
            "n_total_samples": len(combined_train),
            "n_val_samples": len(val_indices),
        },
    }

    # Training loop
    log.info("Starting training: %d epochs, %d train samples, %d val samples",
             args.epochs, len(combined_train) - len(val_indices_from_train), len(val_indices))
    sampler = CurriculumSampler(train_datasets, concat_offsets, val_indices_from_train,
                                total_epochs=args.epochs, auto_label_frac=args.auto_label_frac)
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=8, persistent_workers=True)
    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        sampler.set_epoch(epoch)
        n_train = len(sampler)
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight, epoch)
        val_loss, ious = validate(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - t_epoch
        elapsed = time.time() - t_train_start
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)

        iou_str = "  ".join(f"IoU@{t}={v:.4f}" for t, v in sorted(ious.items()))
        log.info("epoch %3d: train=%.4f  val=%.4f  %s  lr=%.1e  "
                 "samples=%d  %.0fs/ep  ETA=%.0fm",
                 epoch + 1, train_loss, val_loss, iou_str,
                 optimizer.param_groups[0]['lr'],
                 n_train, epoch_time, eta / 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "ious": ious,
                "model_state_dict": model.state_dict(),
                **ckpt_meta,
            }, args.out)
            log.info("  Saved best (val_loss=%.4f) at epoch %d", val_loss, epoch + 1)

        torch.save({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ious": ious,
            "model_state_dict": model.state_dict(),
            **ckpt_meta,
        }, latest_path)

    log.info("Done. Best val_loss=%.4f -> %s", best_val_loss, args.out)

    # Export sidecar
    from sarvalanche.ml.export_weights import export_weights
    export_weights(
        checkpoint_path=args.out,
        model_name="pairwise_debris_detector",
        train_samples=len(combined_train) - len(val_indices_from_train),
        test_samples=len(val_indices),
        extra_metrics=ious,
        notes=f"Trained on {len(zone_names)} zones: {', '.join(zone_names)}",
        weights_dir=args.out.parent,
    )


if __name__ == "__main__":
    main()
