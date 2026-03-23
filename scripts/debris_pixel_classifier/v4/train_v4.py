"""Train v4 multi-scale detector with focal loss + curriculum learning.

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v4/train_v4.py \
        --nc season_*.nc \
        --date 2024-11-15 2024-12-29 ... \
        --polygons labels_*.gpkg ... \
        --epochs 50 --out v4_best.pt
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
from sarvalanche.ml.v3.channels import N_INPUT, N_STATIC
from sarvalanche.ml.v3.patch_extraction import (
    build_static_stack,
    get_pair_metadata_and_tracks,
)
from sarvalanche.ml.v3.dataset_inmemory import build_inmemory_dataset
from sarvalanche.ml.v4.model import MultiScaleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _resolve_device(s):
    if s: return torch.device(s)
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def focal_bce(logits, targets, sample_weights=None, gamma=2.0, alpha=0.75):
    """Focal loss. alpha=0.75 gives 3× weight to positive (debris) class."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce
    if sample_weights is not None:
        return (loss * sample_weights.view(-1, 1, 1, 1)).mean()
    return loss.mean()


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


def train_epoch(model, loader, optimizer, device, epoch=0):
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Train ep{epoch+1}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        fine = batch["x"].to(device)
        local_ctx = batch["local_ctx"].to(device)
        regional = batch["regional"].to(device)
        targets = batch["label"].to(device)
        weights = batch["confidence"].to(device)

        optimizer.zero_grad()
        logits = model(fine, local_ctx, regional)
        loss = focal_bce(logits, targets, weights) + dice_loss(logits, targets, weights)
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
    intersection = {t: 0 for t in thresholds}
    union = {t: 0 for t in thresholds}
    for batch in tqdm(loader, desc="Val", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        fine = batch["x"].to(device)
        local_ctx = batch["local_ctx"].to(device)
        regional = batch["regional"].to(device)
        targets = batch["label"].to(device)
        logits = model(fine, local_ctx, regional)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        total_loss += loss.item()
        n += 1
        probs = torch.sigmoid(logits)
        for t in thresholds:
            preds = (probs >= t).float()
            intersection[t] += (preds * targets).sum().item()
            union[t] += ((preds + targets) >= 1).float().sum().item()
    ious = {t: intersection[t] / max(union[t], 1) for t in thresholds}
    return total_loss / max(n, 1), ious


def main():
    parser = argparse.ArgumentParser(description="Train v4 multi-scale detector")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--date", type=str, nargs="+", required=True)
    parser.add_argument("--polygons", type=Path, nargs="+", required=True)
    parser.add_argument("--geotiff-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--val-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    n_dates = len(args.date)
    assert len(args.polygons) == n_dates
    geotiff_dirs = list(args.geotiff_dir) + [None] * (n_dates - len(args.geotiff_dir))

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Load data
    t0 = time.time()
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("Loaded in %.0fs", time.time() - t0)

    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        import xarray as xr
        hrrr_ds = xr.open_dataset(args.hrrr)

    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    pair_metas, tracks, hrrr_cache = get_pair_metadata_and_tracks(ds, hrrr_ds=hrrr_ds)

    H, W = ds.sizes["y"], ds.sizes["x"]
    val_path_mask = np.zeros((H, W), dtype=bool)

    date_polygon_pairs = []
    for di in range(n_dates):
        gdf = gpd.read_file(args.polygons[di])
        date_polygon_pairs.append((args.date[di], gdf, geotiff_dirs[di]))

    # Build dataset (includes regional precomputation)
    dataset = build_inmemory_dataset(
        ds, pair_metas, tracks, hrrr_cache, static_scene,
        date_polygon_pairs, val_path_mask,
        stride=args.stride, neg_ratio=1.0, augment=True,
    )

    # Split
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

    log.info("Train: %d pos + %d neg, Val: %d", len(train_pos_idx), len(train_neg_idx), len(val_indices))

    n_pos_per_epoch = len(train_pos_idx)
    if args.test_mode:
        n_pos_per_epoch = min(n_pos_per_epoch, 14000)
        log.info("TEST MODE: capping pos to %d", n_pos_per_epoch)
    n_neg_per_epoch = min(n_pos_per_epoch, len(train_neg_idx))

    max_val = 15000
    if len(val_indices) > max_val:
        val_subset = np.random.default_rng(99).choice(val_indices, max_val, replace=False)
    else:
        val_subset = val_indices
    val_ds = Subset(dataset, val_subset)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    # Model
    model = MultiScaleDetector(in_ch=N_INPUT, base_ch=args.base_ch).to(device)
    if args.resume and args.resume.exists():
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        log.info("Resumed from %s", args.resume)
    log.info("Model: %d params (in_ch=%d, base_ch=%d)",
             sum(p.numel() for p in model.parameters()), N_INPUT, args.base_ch)

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            log.info("Model compiled with torch.compile")
        except Exception as e:
            log.warning("torch.compile failed: %s", e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01,
    )

    best_val_loss = float("inf")
    out_path = args.out or Path("v4_best.pt")

    # Sampler
    from torch.utils.data import Sampler

    class EpochSubsampler(Sampler):
        def __init__(self, pos_idx, neg_idx, n_pos, n_neg):
            self.pos_idx, self.neg_idx = pos_idx, neg_idx
            self.n_pos, self.n_neg = n_pos, n_neg
            self.epoch = 0
        def set_epoch(self, e): self.epoch = e
        def __iter__(self):
            rng = np.random.default_rng(self.epoch)
            pos = rng.choice(self.pos_idx, min(self.n_pos, len(self.pos_idx)), replace=False)
            neg = rng.choice(self.neg_idx, min(self.n_neg, len(self.neg_idx)), replace=False)
            all_idx = np.concatenate([pos, neg])
            rng.shuffle(all_idx)
            return iter(all_idx.tolist())
        def __len__(self): return self.n_pos + self.n_neg

    sampler = EpochSubsampler(train_pos_idx, train_neg_idx, n_pos_per_epoch, n_neg_per_epoch)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, persistent_workers=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        dataset.epoch = epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, ious = validate(model, val_loader, device)
        scheduler.step()
        iou_str = "  ".join(f"IoU@{t}={v:.4f}" for t, v in sorted(ious.items()))
        log.info("epoch %3d: train=%.4f  val=%.4f  %s  lr=%.1e",
                 epoch + 1, train_loss, val_loss, iou_str,
                 optimizer.param_groups[0]['lr'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            log.info("  Saved (val_loss=%.4f) at epoch %d", val_loss, epoch + 1)

    log.info("Done. Best val_loss=%.4f → %s", best_val_loss, out_path)
    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
