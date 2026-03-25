"""Train v3 with in-memory dataset — no disk extraction needed.

Loads the season netcdf once, computes pair metadata + static stack,
then generates patches on-the-fly during training.

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/train_inmemory.py \
        --nc season_*.nc \
        --date 2024-11-15 2024-12-29 \
        --polygons labels_2024-11-15.gpkg labels_2024-12-29.gpkg \
        --geotiff-dir geotiffs/2024-11-15 geotiffs/2024-12-29 \
        --val-paths obs_paths.gpkg \
        --epochs 30 --test-mode
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
import torch
torch.set_float32_matmul_precision('medium')

import argparse
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT, N_SAR, N_STATIC, SAR_CHANNELS, STATIC_CHANNELS
from sarvalanche.ml.v3.model import SinglePairDetector
from sarvalanche.ml.v3.patch_extraction import (
    build_static_stack,
    get_pair_metadata_and_tracks,
)
from sarvalanche.ml.v3.dataset_inmemory import build_inmemory_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _resolve_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def weighted_bce(logits, targets, pos_weight, sample_weights=None):
    per_pixel = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none",
    )
    if sample_weights is not None:
        return (per_pixel * sample_weights.view(-1, 1, 1, 1)).mean()
    return per_pixel.mean()


def dice_loss(logits, targets, sample_weights=None, smooth=1.0):
    probs = torch.sigmoid(logits)
    if sample_weights is not None:
        B = logits.shape[0]
        losses = []
        for i in range(B):
            p, t = probs[i].flatten(), targets[i].flatten()
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
        x = batch["x"].to(device)
        targets = batch["label"].to(device)
        weights = batch["confidence"].to(device)

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
        if n % 50 == 0:
            pbar.set_postfix(loss=f"{total_loss/n:.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, thresholds=(0.2, 0.3, 0.5)):
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    n = 0
    # Track IoU at multiple thresholds
    intersection = {t: 0 for t in thresholds}
    union = {t: 0 for t in thresholds}
    max_prob = 0.0
    total_pos_px = 0
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
        max_prob = max(max_prob, probs.max().item())
        total_pos_px += targets.sum().item()
        total_px += targets.numel()
        for t in thresholds:
            preds = (probs >= t).float()
            intersection[t] += (preds * targets).sum().item()
            union[t] += ((preds + targets) >= 1).float().sum().item()
    ious = {t: intersection[t] / max(union[t], 1) for t in thresholds}
    log.info("  val diag: max_prob=%.4f  pos_px=%d/%d (%.4f%%)",
             max_prob, int(total_pos_px), total_px,
             100.0 * total_pos_px / max(total_px, 1))
    return total_loss / max(n, 1), ious


def main():
    parser = argparse.ArgumentParser(description="Train v3 in-memory (no disk extraction)")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--date", type=str, nargs="+", required=True)
    parser.add_argument("--polygons", type=Path, nargs="+", required=True)
    parser.add_argument("--geotiff-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--val-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint weights")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    n_dates = len(args.date)
    assert len(args.polygons) == n_dates
    geotiff_dirs = list(args.geotiff_dir) + [None] * (n_dates - len(args.geotiff_dir))

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # ═══ LOAD DATA ════════════════════════════════════════════════════
    t0 = time.time()
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  Loaded in %.0fs", time.time() - t0)

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        import xarray as xr
        hrrr_ds = xr.open_dataset(args.hrrr)

    # Static stack
    t0 = time.time()
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("  Static stack: %s (%.0fs)", static_scene.shape, time.time() - t0)

    # Pair metadata + tracks
    t0 = time.time()
    pair_metas, tracks, hrrr_cache = get_pair_metadata_and_tracks(ds, hrrr_ds=hrrr_ds)
    log.info("  Pairs: %d (%.0fs)", len(pair_metas), time.time() - t0)

    # Val paths
    H, W = ds.sizes["y"], ds.sizes["x"]
    val_path_mask = np.zeros((H, W), dtype=bool)
    # (simplified — full rasterization in build_inmemory_dataset)

    # Date-polygon pairs
    date_polygon_pairs = []
    for di in range(n_dates):
        gdf = gpd.read_file(args.polygons[di])
        date_polygon_pairs.append((args.date[di], gdf, geotiff_dirs[di]))

    # ═══ BUILD DATASET ════════════════════════════════════════════════
    t0 = time.time()
    log.info("Building in-memory dataset...")
    dataset = build_inmemory_dataset(
        ds, pair_metas, tracks, hrrr_cache, static_scene,
        date_polygon_pairs, val_path_mask,
        stride=args.stride, neg_ratio=1.0, augment=True,
    )
    log.info("  Dataset: %d samples (%.0fs)", len(dataset), time.time() - t0)

    # Split pos/neg for subsampling
    pos_indices = [i for i, l in enumerate(dataset.labels) if l == 1]
    neg_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
    pos_indices = np.array(pos_indices)
    neg_indices = np.array(neg_indices)
    log.info("  Pos: %d, Neg: %d (%.1f%% pos)",
             len(pos_indices), len(neg_indices),
             100 * len(pos_indices) / max(len(dataset), 1))

    # Spatial split by position
    from collections import defaultdict
    pos_to_indices = defaultdict(list)
    for idx, (date_idx, has_debris, y0, x0, pair_idx, conf) in enumerate(dataset.positions):
        key = f"{'pos' if has_debris else 'neg'}_{y0:04d}_{x0:04d}"
        pos_to_indices[key].append(idx)

    positions = list(pos_to_indices.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(positions)
    n_val_pos = max(1, int(len(positions) * args.val_frac))
    val_positions = set(positions[:n_val_pos])
    train_positions = set(positions[n_val_pos:])

    train_indices = [idx for pos in train_positions for idx in pos_to_indices[pos]]
    val_indices = [idx for pos in val_positions for idx in pos_to_indices[pos]]

    train_pos_idx = np.array([i for i in train_indices if dataset.labels[i] == 1])
    train_neg_idx = np.array([i for i in train_indices if dataset.labels[i] == 0])

    log.info("Spatial split: %d train pos (%d pos, %d neg), %d val",
             len(train_positions), len(train_pos_idx), len(train_neg_idx), len(val_indices))

    # Per-epoch subsampling
    n_pos_per_epoch = len(train_pos_idx)
    if args.test_mode:
        max_samples = 28000
        n_pos_per_epoch = min(n_pos_per_epoch, max_samples // 2)
        log.info("TEST MODE: capping pos to %d", n_pos_per_epoch)

    n_neg_per_epoch = min(n_pos_per_epoch, len(train_neg_idx))
    log.info("Per-epoch: %d pos + %d neg = %d", n_pos_per_epoch, n_neg_per_epoch,
             n_pos_per_epoch + n_neg_per_epoch)

    # Val subset
    max_val = 15000
    if len(val_indices) > max_val:
        val_subset = np.random.default_rng(99).choice(val_indices, max_val, replace=False)
    else:
        val_subset = val_indices
    val_ds = Subset(dataset, val_subset)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Pos weight — compute from PIXEL ratio, not patch ratio
    # Positive patches have ~2% debris pixels, so even with 1:1 patch sampling
    # the pixel ratio is ~50:1
    log.info("Sampling pixel ratio from positive patches...")
    n_pos_px = n_total_px = 0
    sample_rng = np.random.default_rng(77)
    sample_idx = sample_rng.choice(pos_indices, size=min(500, len(pos_indices)), replace=False)
    for si in sample_idx:
        batch = dataset[si]
        mask = batch['label'].numpy()
        n_pos_px += mask.sum()
        n_total_px += mask.size
    n_neg_px = n_total_px - n_pos_px
    pw = min(50.0, float(n_neg_px / max(n_pos_px, 1)))
    pos_weight = torch.tensor([pw], device=device)
    log.info("Pos weight: %.1f (from %d pos / %d total pixels in %d patches)",
             pw, int(n_pos_px), int(n_total_px), len(sample_idx))

    # Model
    ckpt_in_ch = N_INPUT
    model = SinglePairDetector(in_ch=ckpt_in_ch, base_ch=args.base_ch).to(device)
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        log.info("Resumed from %s", args.resume)
    log.info("Model: %d params (in_ch=%d)", sum(p.numel() for p in model.parameters()), ckpt_in_ch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Warm restarts synced with curriculum transitions at epochs 10 and 20
    # T_0=10: first cycle is 10 epochs, then doubles each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_loss = float("inf")
    out_path = args.out or Path("v3_inmemory_best.pt")

    # ═══ TRAINING LOOP ════════════════════════════════════════════════
    from torch.utils.data import Sampler

    class EpochSubsampler(Sampler):
        def __init__(self, pos_idx, neg_idx, n_pos, n_neg):
            self.pos_idx, self.neg_idx = pos_idx, neg_idx
            self.n_pos, self.n_neg = n_pos, n_neg
            self.epoch = 0

        def set_epoch(self, e):
            self.epoch = e

        def __iter__(self):
            rng = np.random.default_rng(self.epoch)
            pos = rng.choice(self.pos_idx, min(self.n_pos, len(self.pos_idx)), replace=False)
            neg = rng.choice(self.neg_idx, min(self.n_neg, len(self.neg_idx)), replace=False)
            all_idx = np.concatenate([pos, neg])
            rng.shuffle(all_idx)
            return iter(all_idx.tolist())

        def __len__(self):
            return self.n_pos + self.n_neg

    sampler = EpochSubsampler(train_pos_idx, train_neg_idx, n_pos_per_epoch, n_neg_per_epoch)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, persistent_workers=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        dataset.epoch = epoch  # curriculum learning — controls signal threshold
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight, epoch, scaler)
        val_loss, ious = validate(model, val_loader, device)
        scheduler.step()

        iou_str = "  ".join(f"IoU@{t}={v:.4f}" for t, v in sorted(ious.items()))
        log.info("epoch %3d: train=%.4f  val=%.4f  %s  lr=%.1e",
                 epoch + 1, train_loss, val_loss, iou_str,
                 optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "ious": ious,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "architecture": "SinglePairDetector",
                    "in_ch": ckpt_in_ch,
                    "base_ch": args.base_ch,
                    "n_sar": N_SAR,
                    "n_static": N_STATIC,
                    "sar_channels": SAR_CHANNELS,
                    "static_channels": STATIC_CHANNELS,
                },
                "zones": [str(args.nc)],
                "seasons": args.date,
                "training_args": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "stride": args.stride,
                    "val_frac": args.val_frac,
                    "epochs": args.epochs,
                    "resume": str(args.resume) if args.resume else None,
                    "hrrr": str(args.hrrr) if args.hrrr else None,
                    "pos_weight": pw,
                    "n_train_pos": len(train_pos_idx),
                    "n_train_neg": len(train_neg_idx),
                    "n_val": len(val_subset),
                },
            }, out_path)
            log.info("  Saved best (val_loss=%.4f) at epoch %d", val_loss, epoch + 1)

        # Always save latest checkpoint so we know how far training got
        latest_path = out_path.with_stem(out_path.stem + "_latest")
        torch.save({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ious": ious,
            "model_state_dict": model.state_dict(),
        }, latest_path)

    log.info("Done. Best val_loss=%.4f → %s", best_val_loss, out_path)

    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
