"""Learning curve analysis for v2 debris detection CNN.

Runs leave-one-date-out cross-validation and cumulative learning curve
experiments to assess training set sufficiency, then fits a power law
to extrapolate expected performance with more labeled dates.

Usage:
    # Quick test (~30 min on MPS)
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/learning_curve.py \
        --data-dir local/issw/v2_patches --n-seeds 1 --epochs 30

    # Full run (~2 hours on MPS)
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/learning_curve.py \
        --data-dir local/issw/v2_patches --n-seeds 3 --epochs 30
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from torch.utils.data import ConcatDataset, DataLoader

from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import DebrisDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_date_datasets(data_dir: Path) -> dict[str, V2PatchDataset]:
    """Discover date subdirectories and return one V2PatchDataset per date."""
    datasets: dict[str, V2PatchDataset] = {}
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ds = V2PatchDataset(subdir, augment=False)
        if len(ds) > 0:
            datasets[subdir.name] = ds
            log.info("Date %s: %d patches", subdir.name, len(ds))
    if not datasets:
        raise FileNotFoundError(f"No date subdirectories with patches found in {data_dir}")
    return datasets


def compute_pos_weight(datasets: dict[str, V2PatchDataset], date_keys: list[str]) -> float:
    """Compute pixel-level pos/neg ratio across the given dates."""
    n_pos = 0
    n_total = 0
    for key in date_keys:
        ds = datasets[key]
        for i in range(len(ds)):
            data = np.load(ds.files[i], allow_pickle=True)
            if "label_mask" in data:
                mask = data["label_mask"]
                n_pos += mask.sum()
                n_total += mask.size
            else:
                label = int(data["label"])
                n_pos += label * 128 * 128
                n_total += 128 * 128
    n_neg = n_total - n_pos
    pw = float(n_neg / max(n_pos, 1))
    return min(pw, 50.0)


def _weighted_bce(logits, targets, pos_weight):
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight,
    )


def _train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        sar_maps = [m.to(device) for m in batch["sar_maps"]]
        static = batch["static"].to(device)
        targets = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(sar_maps, static)
        loss = _weighted_bce(logits, targets, pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate(model, loader, device, threshold=0.5):
    model.eval()
    intersection = 0
    union = 0
    tp = 0
    fp = 0
    fn = 0
    for batch in loader:
        sar_maps = [m.to(device) for m in batch["sar_maps"]]
        static = batch["static"].to(device)
        targets = batch["label"].to(device)
        logits = model(sar_maps, static)
        preds = (torch.sigmoid(logits) >= threshold).float()
        tp += (preds * targets).sum().item()
        fp += (preds * (1 - targets)).sum().item()
        fn += ((1 - preds) * targets).sum().item()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()
    iou = intersection / max(union, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"iou": iou, "precision": precision, "recall": recall}


def train_and_evaluate(
    datasets: dict[str, V2PatchDataset],
    train_dates: list[str],
    test_dates: list[str],
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Train a fresh model on train_dates, evaluate on test_dates."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build train/test loaders
    train_datasets = []
    for key in train_dates:
        ds = V2PatchDataset(datasets[key].data_dir, augment=True)
        train_datasets.append(ds)
    train_combined = ConcatDataset(train_datasets)

    test_datasets = [datasets[key] for key in test_dates]
    test_combined = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=v2_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_combined,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=v2_collate_fn,
        num_workers=0,
    )

    # Pos weight from training data
    pw = compute_pos_weight(datasets, train_dates)
    pos_weight = torch.tensor([pw], device=device)

    # Fresh model
    model = DebrisDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        _train_epoch(model, train_loader, optimizer, device, pos_weight)
        scheduler.step()

    metrics = _evaluate(model, test_loader, device)
    return metrics


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_lodo_cv(
    datasets: dict[str, V2PatchDataset],
    epochs: int,
    batch_size: int,
    device: torch.device,
    n_seeds: int,
) -> pd.DataFrame:
    """Leave-one-date-out cross-validation."""
    all_dates = sorted(datasets.keys())
    rows = []
    total_runs = len(all_dates) * n_seeds
    run_i = 0

    for test_date in all_dates:
        train_dates = [d for d in all_dates if d != test_date]
        for seed in range(n_seeds):
            run_i += 1
            log.info(
                "LODO [%d/%d]: test=%s, train=%s, seed=%d",
                run_i, total_runs, test_date, train_dates, seed,
            )
            metrics = train_and_evaluate(
                datasets, train_dates, [test_date], epochs, batch_size, device, seed,
            )
            rows.append({
                "test_date": test_date,
                "train_dates": ",".join(train_dates),
                "n_train_dates": len(train_dates),
                "seed": seed,
                **metrics,
            })
            log.info("  IoU=%.4f  Prec=%.4f  Rec=%.4f", metrics["iou"], metrics["precision"], metrics["recall"])

    return pd.DataFrame(rows)


def run_learning_curve(
    datasets: dict[str, V2PatchDataset],
    epochs: int,
    batch_size: int,
    device: torch.device,
    n_seeds: int,
) -> pd.DataFrame:
    """Cumulative learning curve: for each test date, train on N={1,...,K-1} dates."""
    all_dates = sorted(datasets.keys())
    rows = []
    total_runs = 0

    # Pre-count runs for logging
    for test_date in all_dates:
        remaining = [d for d in all_dates if d != test_date]
        for n in range(1, len(remaining) + 1):
            total_runs += len(list(itertools.combinations(remaining, n))) * n_seeds

    run_i = 0
    for test_date in all_dates:
        remaining = [d for d in all_dates if d != test_date]
        for n in range(1, len(remaining) + 1):
            for combo in itertools.combinations(remaining, n):
                train_dates = list(combo)
                for seed in range(n_seeds):
                    run_i += 1
                    log.info(
                        "LearningCurve [%d/%d]: test=%s, train=%s (N=%d), seed=%d",
                        run_i, total_runs, test_date, train_dates, n, seed,
                    )
                    metrics = train_and_evaluate(
                        datasets, train_dates, [test_date], epochs, batch_size, device, seed,
                    )
                    rows.append({
                        "test_date": test_date,
                        "train_dates": ",".join(train_dates),
                        "n_train_dates": n,
                        "seed": seed,
                        **metrics,
                    })
                    log.info("  IoU=%.4f", metrics["iou"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Power law fitting
# ---------------------------------------------------------------------------


def _power_law(n, a, b, c):
    """IoU = a - b * N^(-c)"""
    return a - b * np.power(n, -c)


def fit_power_law(df_curve: pd.DataFrame) -> dict:
    """Fit power law to mean IoU vs N training dates."""
    grouped = df_curve.groupby("n_train_dates")["iou"].agg(["mean", "std"]).reset_index()
    n_vals = grouped["n_train_dates"].values.astype(float)
    iou_means = grouped["mean"].values
    iou_stds = grouped["std"].values

    try:
        popt, pcov = curve_fit(
            _power_law, n_vals, iou_means,
            p0=[0.5, 0.3, 0.5],
            bounds=([0, 0, 0.01], [1, 2, 5]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        extrapolated = {
            int(n): float(_power_law(n, *popt))
            for n in [5, 6, 8, 10]
        }
        return {
            "a_asymptote": float(popt[0]),
            "b": float(popt[1]),
            "c": float(popt[2]),
            "a_std": float(perr[0]),
            "b_std": float(perr[1]),
            "c_std": float(perr[2]),
            "n_values": n_vals.tolist(),
            "iou_means": iou_means.tolist(),
            "iou_stds": iou_stds.tolist(),
            "extrapolated": extrapolated,
            "fit_success": True,
        }
    except (RuntimeError, ValueError) as e:
        log.warning("Power law fit failed: %s", e)
        return {
            "n_values": n_vals.tolist(),
            "iou_means": iou_means.tolist(),
            "iou_stds": iou_stds.tolist(),
            "fit_success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lodo_bar(df_lodo: pd.DataFrame, datasets: dict[str, V2PatchDataset], out_dir: Path):
    """Bar chart of per-date IoU from LODO cross-validation."""
    grouped = df_lodo.groupby("test_date")["iou"].agg(["mean", "std"]).reset_index()
    grouped = grouped.sort_values("test_date")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(grouped))
    bars = ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, alpha=0.8, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["test_date"], rotation=30, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("Leave-One-Date-Out Cross-Validation")

    # Annotate with patch counts
    for i, (_, row) in enumerate(grouped.iterrows()):
        date = row["test_date"]
        n_patches = len(datasets[date]) if date in datasets else 0
        n_pos = sum(datasets[date].labels) if date in datasets else 0
        ax.annotate(
            f"n={n_patches}\npos={n_pos}",
            (i, row["mean"] + row["std"] + 0.01),
            ha="center", fontsize=8,
        )

    ax.set_ylim(0, min(1.0, grouped["mean"].max() + grouped["std"].max() + 0.1))
    fig.tight_layout()
    fig.savefig(out_dir / "lodo_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_dir / "lodo_bar.png")


def plot_learning_curve(df_curve: pd.DataFrame, power_law_params: dict, out_dir: Path):
    """IoU vs N training dates with power law fit."""
    grouped = df_curve.groupby("n_train_dates")["iou"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        grouped["n_train_dates"], grouped["mean"], yerr=grouped["std"],
        fmt="o-", capsize=5, markersize=8, color="steelblue", label="Observed",
    )

    if power_law_params.get("fit_success"):
        n_ext = np.linspace(1, 10, 100)
        a, b, c = power_law_params["a_asymptote"], power_law_params["b"], power_law_params["c"]
        iou_ext = _power_law(n_ext, a, b, c)
        ax.plot(n_ext, iou_ext, "--", color="tomato", label=f"Power law (asymptote={a:.3f})")

        for n_pred in [5, 6, 8, 10]:
            iou_pred = power_law_params["extrapolated"][n_pred]
            ax.plot(n_pred, iou_pred, "x", color="tomato", markersize=10)
            ax.annotate(f"N={n_pred}: {iou_pred:.3f}", (n_pred, iou_pred),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Number of Training Dates")
    ax.set_ylabel("IoU")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_dir / "learning_curve.png")


def plot_extrapolation(power_law_params: dict, out_dir: Path):
    """Power law extrapolation with confidence band."""
    if not power_law_params.get("fit_success"):
        log.warning("Skipping extrapolation plot (fit failed)")
        return

    a, b, c = power_law_params["a_asymptote"], power_law_params["b"], power_law_params["c"]
    a_std = power_law_params["a_std"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Observed points
    n_obs = power_law_params["n_values"]
    iou_obs = power_law_params["iou_means"]
    iou_std_obs = power_law_params["iou_stds"]
    ax.errorbar(n_obs, iou_obs, yerr=iou_std_obs, fmt="o", capsize=5,
                markersize=8, color="steelblue", label="Observed")

    # Fit + extrapolation
    n_ext = np.linspace(1, 10, 200)
    iou_fit = _power_law(n_ext, a, b, c)
    iou_upper = _power_law(n_ext, a + a_std, b, c)
    iou_lower = _power_law(n_ext, a - a_std, b, c)

    ax.plot(n_ext, iou_fit, "-", color="tomato", label="Power law fit")
    ax.fill_between(n_ext, iou_lower, iou_upper, alpha=0.2, color="tomato", label="Confidence band")

    # Annotate key extrapolation points
    for n_pred in [5, 10]:
        iou_pred = power_law_params["extrapolated"][n_pred]
        ax.axvline(n_pred, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(
            f"N={n_pred}: IoU={iou_pred:.3f}",
            (n_pred, iou_pred), textcoords="offset points", xytext=(10, -15),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Asymptote
    ax.axhline(a, color="green", linestyle="--", alpha=0.5, label=f"Asymptote = {a:.3f}")

    ax.set_xlabel("Number of Training Dates")
    ax.set_ylabel("IoU")
    ax.set_title("Power Law Extrapolation")
    ax.legend(loc="lower right")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "extrapolation.png", dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_dir / "extrapolation.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    df_lodo: pd.DataFrame,
    df_curve: pd.DataFrame,
    power_law_params: dict,
    out_dir: Path,
):
    """Write text summary with analysis and recommendations."""
    lines = []
    lines.append("=" * 60)
    lines.append("LEARNING CURVE ANALYSIS SUMMARY")
    lines.append("=" * 60)

    # LODO results
    lines.append("\n--- Leave-One-Date-Out Cross-Validation ---")
    lodo_mean = df_lodo["iou"].mean()
    lodo_std = df_lodo["iou"].std()
    lines.append(f"Overall IoU: {lodo_mean:.4f} +/- {lodo_std:.4f}")

    per_date = df_lodo.groupby("test_date")["iou"].agg(["mean", "std"])
    hardest = per_date["mean"].idxmin()
    easiest = per_date["mean"].idxmax()
    lines.append(f"Hardest test date: {hardest} (IoU={per_date.loc[hardest, 'mean']:.4f})")
    lines.append(f"Easiest test date: {easiest} (IoU={per_date.loc[easiest, 'mean']:.4f})")

    lines.append("\nPer-date results:")
    for date, row in per_date.iterrows():
        lines.append(f"  {date}: IoU={row['mean']:.4f} +/- {row['std']:.4f}")

    # Learning curve marginal gains
    lines.append("\n--- Learning Curve: Marginal Gains ---")
    grouped = df_curve.groupby("n_train_dates")["iou"].mean()
    prev = None
    for n in sorted(grouped.index):
        iou = grouped[n]
        if prev is not None:
            gain = iou - prev
            lines.append(f"N={int(n-1)}->{int(n)}: IoU gain = {gain:+.4f} (mean IoU={iou:.4f})")
        else:
            lines.append(f"N={int(n)}: mean IoU={iou:.4f}")
        prev = iou

    # Power law extrapolation
    lines.append("\n--- Power Law Extrapolation ---")
    if power_law_params.get("fit_success"):
        a = power_law_params["a_asymptote"]
        a_std = power_law_params["a_std"]
        lines.append(f"Saturation IoU (asymptote): {a:.4f} +/- {a_std:.4f}")
        lines.append(f"Model: IoU = {a:.4f} - {power_law_params['b']:.4f} * N^(-{power_law_params['c']:.4f})")
        for n, iou in power_law_params["extrapolated"].items():
            lines.append(f"  Predicted IoU at N={n}: {iou:.4f}")

        # Recommendation
        lines.append("\n--- Recommendation ---")
        last_gain = None
        ns = sorted(grouped.index)
        if len(ns) >= 2:
            last_gain = grouped[ns[-1]] - grouped[ns[-2]]

        if last_gain is not None and last_gain < 0.01:
            lines.append("Status: APPROACHING SATURATION")
            lines.append(f"  Last marginal gain ({last_gain:+.4f}) is small.")
            lines.append("  Additional dates may provide diminishing returns.")
        elif last_gain is not None and last_gain > 0.03:
            lines.append("Status: DATA-LIMITED")
            lines.append(f"  Last marginal gain ({last_gain:+.4f}) is substantial.")
            lines.append("  More labeled dates would likely improve performance.")
        else:
            lines.append("Status: MODERATE GAINS")
            if last_gain is not None:
                lines.append(f"  Last marginal gain ({last_gain:+.4f}).")
            lines.append("  Additional dates may help but gains are tapering.")
    else:
        lines.append(f"Power law fit FAILED: {power_law_params.get('error', 'unknown')}")
        lines.append("Cannot extrapolate. Consider collecting more data points.")

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary)
    log.info("Saved %s", out_dir / "summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Learning curve analysis for v2 debris detector")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to v2_patches/ with date subdirs")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--n-seeds", type=int, default=3, help="Random seeds per experiment")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: data-dir/learning_curve/)")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    log.info("Using device: %s", device)

    out_dir = args.out_dir or (args.data_dir / "learning_curve")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets per date
    datasets = load_date_datasets(args.data_dir)
    log.info("Found %d dates: %s", len(datasets), sorted(datasets.keys()))

    if len(datasets) < 2:
        log.error("Need at least 2 dates for cross-validation, found %d", len(datasets))
        return

    # Experiment 1: LODO
    log.info("=" * 60)
    log.info("EXPERIMENT 1: Leave-One-Date-Out Cross-Validation")
    log.info("=" * 60)
    df_lodo = run_lodo_cv(datasets, args.epochs, args.batch_size, device, args.n_seeds)
    df_lodo.to_csv(out_dir / "lodo_results.csv", index=False)
    log.info("Saved LODO results to %s", out_dir / "lodo_results.csv")

    # Experiment 2: Cumulative learning curve
    log.info("=" * 60)
    log.info("EXPERIMENT 2: Cumulative Learning Curve")
    log.info("=" * 60)
    df_curve = run_learning_curve(datasets, args.epochs, args.batch_size, device, args.n_seeds)
    df_curve.to_csv(out_dir / "learning_curve_results.csv", index=False)
    log.info("Saved learning curve results to %s", out_dir / "learning_curve_results.csv")

    # Experiment 3: Power law fit
    log.info("=" * 60)
    log.info("EXPERIMENT 3: Power Law Extrapolation")
    log.info("=" * 60)
    power_law_params = fit_power_law(df_curve)
    with open(out_dir / "power_law_params.json", "w") as f:
        json.dump(power_law_params, f, indent=2)
    log.info("Saved power law params to %s", out_dir / "power_law_params.json")

    # Plots
    log.info("Generating plots...")
    plot_lodo_bar(df_lodo, datasets, out_dir)
    plot_learning_curve(df_curve, power_law_params, out_dir)
    plot_extrapolation(power_law_params, out_dir)

    # Summary
    print_summary(df_lodo, df_curve, power_law_params, out_dir)


if __name__ == "__main__":
    main()
