"""Run training experiments comparing human-only, auto-only, combined, and pretrain+finetune.

Splits human-labeled dates into train/test by confidence, then runs 4 experiments:
  0. human-only (baseline)       — train on human-train, eval on human-test
  1. auto-only                   — train on auto, eval on human-test
  2. combined                    — train on auto + human-train, eval on human-test
  3. pretrain auto → finetune    — pretrain on auto, finetune on human-train, eval on human-test

Human test dates (high confidence, diverse):
  - 2025-02-04 (254 pos, conf 0.658, mid-winter)
  - 2024-02-04 (279 pos, galena, mid-winter, different year)
  - 2025-03-15 (143 pos, conf 0.643, spring)

Excluded from training (low confidence <0.4):
  - 2021-12-26 (conf 0.339)
  - 2022-12-01 (conf 0.355)

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/run_training_experiments.py \
        --human-patches local/issw/v2_patches \
        --auto-patches local/debris_shapes/CNFAIC_auto_patches \
        --out-dir local/experiments/auto_vs_human \
        --epochs 50 --batch-size 4
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Date-level split by confidence analysis
TEST_DATES = ["2025-02-04", "2024-02-04", "2025-03-15"]
EXCLUDE_DATES = ["2021-12-26", "2022-12-01"]


def find_date_dirs(patches_root: Path) -> dict[str, list[Path]]:
    """Map date strings to directories containing labels.json."""
    date_dirs: dict[str, list[Path]] = {}
    for lf in sorted(patches_root.rglob("labels.json")):
        # Extract date from path components
        for part in lf.parent.parts:
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                date_dirs.setdefault(part, []).append(lf.parent)
                break
    return date_dirs


def link_date_dirs(date_dirs: dict[str, list[Path]], dates: list[str], dest: Path):
    """Create real subdirectories with labels.json + symlinked npz files.

    Path.rglob() doesn't follow directory symlinks, so we create real
    directories and symlink individual files instead.
    """
    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    for date in dates:
        if date not in date_dirs:
            continue
        for src_dir in date_dirs[date]:
            # Create unique name from relative path
            link_name = src_dir.name
            if len(date_dirs[date]) > 1:
                link_name = f"{src_dir.parent.name}_{src_dir.name}"
            sub_dir = dest / link_name
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
            sub_dir.mkdir(parents=True)

            # Copy labels.json (small file)
            src_labels = src_dir / "labels.json"
            if src_labels.exists():
                shutil.copy2(src_labels, sub_dir / "labels.json")

            # Symlink individual npz files
            for npz in src_dir.glob("*_v2_*.npz"):
                (sub_dir / npz.name).symlink_to(npz.resolve())

            linked += 1
    return linked


def count_patches(data_dir: Path) -> tuple[int, int]:
    """Count pos/neg patches in a directory tree."""
    n_pos = n_neg = 0
    for lf in data_dir.rglob("labels.json"):
        with open(lf) as f:
            labels = json.load(f)
        for v in labels.values():
            if v.get("label") == 1:
                n_pos += 1
            elif v.get("label") == 0:
                n_neg += 1
    return n_pos, n_neg


def run_train(data_dir: Path, out_weights: Path, args, pretrained_weights: Path = None,
              epochs: int = None, lr: float = None):
    """Run training and return the command's exit code."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/debris_pixel_classifier/v2/train.py",
        "--data-dir", str(data_dir),
        "--epochs", str(epochs or args.epochs),
        "--lr", str(lr or args.lr),
        "--batch-size", str(args.batch_size),
        "--out", str(out_weights),
    ]
    if args.pos_weight > 0:
        cmd += ["--pos-weight", str(args.pos_weight)]
    if args.device:
        cmd += ["--device", args.device]
    if args.skip:
        cmd += ["--skip"]
    if hasattr(args, 'sar_channels') and args.sar_channels != 2:
        cmd += ["--sar-channels", str(args.sar_channels)]

    log.info("  CMD: %s", " ".join(cmd))

    if pretrained_weights and pretrained_weights.exists():
        cmd += ["--resume", str(pretrained_weights)]
        log.info("  Initialized from pretrained: %s", pretrained_weights)

    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_eval(weights: Path, data_dir: Path, out_json: Path, args):
    """Evaluate a model on a dataset and save metrics."""
    import subprocess
    import sys

    # Use a simple eval script inline via python -c
    eval_code = f"""
import json, logging, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import DebrisDetector, DebrisDetectorSkip

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = torch.device('{args.device or "mps" if not args.device else args.device}')
if not torch.backends.mps.is_available() and '{args.device}' == 'None':
    device = torch.device('cpu')

data_dir = Path('{data_dir}')
ds = V2PatchDataset(data_dir, augment=False)
if len(ds) == 0:
    print(json.dumps({{"error": "no patches"}}))
    exit(0)

loader = DataLoader(ds, batch_size={args.batch_size}, shuffle=False,
                    collate_fn=v2_collate_fn, num_workers=0)

sar_ch = {getattr(args, 'sar_channels', 2)}
model_cls = DebrisDetectorSkip if {args.skip} else DebrisDetector
model = model_cls(sar_in_ch=sar_ch).to(device)
model.load_state_dict(torch.load('{weights}', map_location=device, weights_only=True))
model.eval()

total_loss = 0.0
n_batches = 0
intersection = 0
union = 0
tp = fp = fn = tn = 0

with torch.no_grad():
    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)
        logits = model(sar_maps, static)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        total_loss += loss.item()
        n_batches += 1
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp += (preds * targets).sum().item()
        fp += (preds * (1 - targets)).sum().item()
        fn += ((1 - preds) * targets).sum().item()
        tn += ((1 - preds) * (1 - targets)).sum().item()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()

val_loss = total_loss / max(n_batches, 1)
iou = intersection / max(union, 1)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-6)

metrics = {{
    'val_loss': round(val_loss, 4),
    'iou': round(iou, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1': round(f1, 4),
    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    'n_patches': len(ds),
}}
log.info('Metrics: %s', json.dumps(metrics, indent=2))
with open('{out_json}', 'w') as f:
    json.dump(metrics, f, indent=2)
"""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-c", eval_code],
        check=False,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run auto vs human label training experiments",
    )
    parser.add_argument("--human-patches", type=Path, required=True,
                        help="Root directory of human-labeled v2 patches")
    parser.add_argument("--auto-patches", type=Path, default=None,
                        help="Root directory of auto-labeled v2 patches (if already extracted)")
    parser.add_argument("--out-dir", type=Path, default=Path("local/experiments/auto_vs_human"),
                        help="Output directory for experiment results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip", action="store_true", help="Use DebrisDetectorSkip")
    parser.add_argument("--finetune-epochs", type=int, default=20,
                        help="Epochs for finetune phase (default: 20)")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate for finetune phase (default: 1e-4)")
    parser.add_argument("--sar-channels", type=int, default=2,
                        help="SAR input channels (2=change+ANF, 3=change+ANF+proximity)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover and split human dates ────────────────────────────────
    human_dates = find_date_dirs(args.human_patches)
    all_human_dates = sorted(human_dates.keys())
    log.info("Human-labeled dates: %s", all_human_dates)

    train_dates = [d for d in all_human_dates if d not in TEST_DATES and d not in EXCLUDE_DATES]
    test_dates = [d for d in TEST_DATES if d in human_dates]
    excluded = [d for d in EXCLUDE_DATES if d in human_dates]

    log.info("Train dates (%d): %s", len(train_dates), train_dates)
    log.info("Test dates  (%d): %s", len(test_dates), test_dates)
    log.info("Excluded    (%d): %s", len(excluded), excluded)

    # ── Build symlink trees ───────────────────────────────────────────
    human_train_dir = args.out_dir / "splits" / "human_train"
    human_test_dir = args.out_dir / "splits" / "human_test"

    n = link_date_dirs(human_dates, train_dates, human_train_dir)
    log.info("Linked %d human train dirs", n)
    n = link_date_dirs(human_dates, test_dates, human_test_dir)
    log.info("Linked %d human test dirs", n)

    # Combined = auto + human-train
    combined_dir = args.out_dir / "splits" / "combined"
    n = link_date_dirs(human_dates, train_dates, combined_dir)
    if args.auto_patches:
        auto_dates = find_date_dirs(args.auto_patches)
        # Add auto patches into combined with "auto_" prefix to avoid collision
        auto_as_one = {}
        for date, dirs in auto_dates.items():
            for src_dir in dirs:
                key = f"auto_{src_dir.parent.name}_{date}"
                auto_as_one.setdefault(key, []).append(src_dir)
        for key, dirs in auto_as_one.items():
            for src_dir in dirs:
                sub = combined_dir / f"auto_{src_dir.parent.name}_{src_dir.name}"
                if sub.exists():
                    shutil.rmtree(sub)
                sub.mkdir(parents=True)
                src_labels = src_dir / "labels.json"
                if src_labels.exists():
                    shutil.copy2(src_labels, sub / "labels.json")
                for npz in src_dir.glob("*_v2_*.npz"):
                    (sub / npz.name).symlink_to(npz.resolve())
                n += 1
        # Also create auto-only dir
        auto_only_dir = args.out_dir / "splits" / "auto_only"
        link_date_dirs(auto_dates, sorted(auto_dates.keys()), auto_only_dir)
    else:
        auto_only_dir = None

    # Report patch counts
    train_pos, train_neg = count_patches(human_train_dir)
    test_pos, test_neg = count_patches(human_test_dir)
    log.info("Human train: %d pos, %d neg", train_pos, train_neg)
    log.info("Human test:  %d pos, %d neg", test_pos, test_neg)
    if args.auto_patches:
        auto_pos, auto_neg = count_patches(auto_only_dir)
        comb_pos, comb_neg = count_patches(combined_dir)
        log.info("Auto only:   %d pos, %d neg", auto_pos, auto_neg)
        log.info("Combined:    %d pos, %d neg", comb_pos, comb_neg)

    # ── Save split info ───────────────────────────────────────────────
    split_info = {
        "test_dates": test_dates,
        "train_dates": train_dates,
        "excluded_dates": excluded,
        "human_train_patches": {"pos": train_pos, "neg": train_neg},
        "human_test_patches": {"pos": test_pos, "neg": test_neg},
    }
    with open(args.out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ── Experiment 0: Human-only baseline ─────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("EXPERIMENT 0: Human-only (baseline)")
    log.info("=" * 60)
    exp0_weights = args.out_dir / "exp0_human_only" / "best.pt"
    exp0_weights.parent.mkdir(parents=True, exist_ok=True)
    run_train(human_train_dir, exp0_weights, args)
    exp0_metrics = args.out_dir / "exp0_human_only" / "test_metrics.json"
    run_eval(exp0_weights, human_test_dir, exp0_metrics, args)

    # ── Experiment 1: Auto-only ───────────────────────────────────────
    if auto_only_dir:
        log.info("")
        log.info("=" * 60)
        log.info("EXPERIMENT 1: Auto-only")
        log.info("=" * 60)
        exp1_weights = args.out_dir / "exp1_auto_only" / "best.pt"
        exp1_weights.parent.mkdir(parents=True, exist_ok=True)
        run_train(auto_only_dir, exp1_weights, args)
        exp1_metrics = args.out_dir / "exp1_auto_only" / "test_metrics.json"
        run_eval(exp1_weights, human_test_dir, exp1_metrics, args)

    # ── Experiment 2: Combined ────────────────────────────────────────
    if args.auto_patches:
        log.info("")
        log.info("=" * 60)
        log.info("EXPERIMENT 2: Combined (auto + human-train)")
        log.info("=" * 60)
        exp2_weights = args.out_dir / "exp2_combined" / "best.pt"
        exp2_weights.parent.mkdir(parents=True, exist_ok=True)
        run_train(combined_dir, exp2_weights, args)
        exp2_metrics = args.out_dir / "exp2_combined" / "test_metrics.json"
        run_eval(exp2_weights, human_test_dir, exp2_metrics, args)

    # ── Experiment 3: Pretrain auto → finetune human ──────────────────
    if auto_only_dir:
        log.info("")
        log.info("=" * 60)
        log.info("EXPERIMENT 3: Pretrain auto → Finetune human-train")
        log.info("=" * 60)
        exp3_dir = args.out_dir / "exp3_pretrain_finetune"
        exp3_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: pretrain on auto
        pretrain_weights = exp3_dir / "pretrain.pt"
        log.info("Phase 1: Pretrain on auto labels")
        run_train(auto_only_dir, pretrain_weights, args)

        # Phase 2: finetune on human-train with lower LR
        finetune_weights = exp3_dir / "best.pt"
        log.info("Phase 2: Finetune on human-train (lr=%g, epochs=%d)",
                 args.finetune_lr, args.finetune_epochs)
        run_train(human_train_dir, finetune_weights, args,
                  pretrained_weights=pretrain_weights,
                  epochs=args.finetune_epochs, lr=args.finetune_lr)

        exp3_metrics = exp3_dir / "test_metrics.json"
        run_eval(finetune_weights, human_test_dir, exp3_metrics, args)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)

    for exp_name in ["exp0_human_only", "exp1_auto_only", "exp2_combined", "exp3_pretrain_finetune"]:
        metrics_path = args.out_dir / exp_name / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            log.info(
                "  %-30s  loss=%.4f  IoU=%.4f  F1=%.4f  P=%.4f  R=%.4f",
                exp_name, m["val_loss"], m["iou"], m["f1"], m["precision"], m["recall"],
            )
        else:
            log.info("  %-30s  (not run)", exp_name)


if __name__ == "__main__":
    main()
