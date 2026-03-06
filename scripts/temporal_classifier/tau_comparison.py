"""Compare temporal onset metrics across different tau values.

Reads CNN probability cubes for each tau, runs temporal onset detection,
and outputs a summary CSV + comparison plots.

Usage:
    conda run -n sarvalanche python scripts/temporal_classifier/tau_comparison.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from temporal_onset import run_temporal_onset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path("local/issw/tau_testing")
SAR_NC = Path("local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc")
THRESHOLD = 0.5

TAU_CONFIGS = {
    6: BASE_DIR / "tau6" / "season_v2_debris_probabilities.nc",
    12: BASE_DIR / "tau12" / "season_v2_debris_probabilities.nc",
    18: BASE_DIR / "tau18" / "season_v2_debris_probabilities.nc",
    24: BASE_DIR / "tau24" / "season_v2_debris_probabilities.nc",
    32: BASE_DIR / "tau32" / "season_v2_debris_probabilities.nc",
}


def run_all():
    # Load SAR once
    log.info("Loading SAR dataset...")
    sar_ds = load_netcdf_to_dataset(str(SAR_NC))
    if not np.issubdtype(sar_ds["time"].dtype, np.datetime64):
        sar_ds["time"] = pd.DatetimeIndex(sar_ds["time"].values)
    if any(var.chunks is not None for var in sar_ds.variables.values()):
        sar_ds = sar_ds.load()

    rows = []

    for tau, cnn_path in sorted(TAU_CONFIGS.items()):
        if not cnn_path.exists():
            log.warning("Skipping tau=%d -- %s not found", tau, cnn_path)
            continue

        log.info("=" * 60)
        log.info("Processing tau=%d", tau)

        cnn_ds = xr.open_dataset(cnn_path)
        if not np.issubdtype(cnn_ds["time"].dtype, np.datetime64):
            cnn_ds["time"] = pd.DatetimeIndex(cnn_ds["time"].values)

        result = run_temporal_onset(
            cnn_ds, sar_ds,
            threshold=THRESHOLD,
            pre_existing_max_idx=1,
            min_bump_width=2,
            spatial_radius_px=3,
        )

        # Save onset result
        out_nc = BASE_DIR / f"tau{tau}" / "temporal_onset.nc"
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(out_nc)

        # Extract metrics
        cm = result["candidate_mask"].values
        n_cand = cm.sum()
        spike = result["spike_flag"].values[cm]
        n_spike = spike.sum()

        row = {
            "tau": tau,
            "candidates": int(n_cand),
            "detections": int(n_cand - n_spike),
            "spikes": int(n_spike),
            "spike_pct": 100 * n_spike / max(n_cand, 1),
            "mean_bump_width": float(result["bump_width"].values[cm].mean()),
            "mean_n_above": float(np.nanmean(result["n_above_threshold"].values[cm])),
            "mean_det_prob": float(np.nanmean(result["mean_detection_prob"].values[cm])),
            "mean_persistence": float(np.nanmean(result["persistence_ratio"].values[cm])),
            "mean_bump_smoothness": float(np.nanmean(result["bump_smoothness"].values[cm])),
            "mean_confidence": float(np.nanmean(result["confidence"].values[cm])),
            "mean_peak_prob": float(np.nanmean(result["peak_prob"].values[cm])),
            "mean_spatial_amplitude": float(np.nanmean(result["spatial_bump_amplitude"].values[cm])),
            "mean_spatial_alignment": float(np.nanmean(result["spatial_peak_alignment"].values[cm])),
            "mean_spatial_symmetry": float(np.nanmean(result["spatial_bump_symmetry"].values[cm])),
            "mean_vv_step": float(np.nanmean(np.abs(result["step_height_vv"].values[cm]))),
            "pre_existing": int(result["pre_existing"].values[cm].sum()),
        }
        rows.append(row)
        log.info("tau=%d: %d candidates, %d detections, %d spikes (%.1f%%)",
                 tau, row["candidates"], row["detections"], row["spikes"], row["spike_pct"])

        cnn_ds.close()

    df = pd.DataFrame(rows)
    csv_path = BASE_DIR / "tau_comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV: %s", csv_path)
    print("\n" + df.to_string(index=False))

    return df


def make_plots(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Temporal Onset Metrics vs Tau", fontsize=14, fontweight="bold")

    tau = df["tau"]

    # 1. Spike rate
    ax = axes[0, 0]
    ax.plot(tau, df["spike_pct"], "o-", color="C3", linewidth=2)
    ax.set_ylabel("Spike rate (%)")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Spike Rate")
    ax.grid(True, alpha=0.3)

    # 2. Multi-pass confirmation: width + n_above
    ax = axes[0, 1]
    ax.plot(tau, df["mean_bump_width"], "o-", color="C0", linewidth=2, label="Contiguous width")
    ax.plot(tau, df["mean_n_above"], "s-", color="C2", linewidth=2, label="Total above thresh")
    ax.set_ylabel("Steps (independent passes)")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Multi-Pass Confirmation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Persistence + mean detection prob
    ax = axes[0, 2]
    ax.plot(tau, df["mean_persistence"], "o-", color="C0", linewidth=2, label="Persistence ratio")
    ax.plot(tau, df["mean_det_prob"], "s-", color="C1", linewidth=2, label="Mean detection prob")
    ax.set_ylabel("Score")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Persistence & Detection Consistency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Confidence
    ax = axes[1, 0]
    ax.plot(tau, df["mean_confidence"], "o-", color="C4", linewidth=2)
    ax.set_ylabel("Mean confidence")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Confidence")
    ax.grid(True, alpha=0.3)

    # 5. Spatial metrics
    ax = axes[1, 1]
    ax.plot(tau, df["mean_spatial_amplitude"], "o-", label="Amplitude", linewidth=2)
    ax.plot(tau, df["mean_spatial_alignment"], "s-", label="Peak alignment", linewidth=2)
    ax.plot(tau, df["mean_spatial_symmetry"], "^-", label="Symmetry", linewidth=2)
    ax.set_ylabel("Score")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Spatial Metrics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Detection counts
    ax = axes[1, 2]
    ax.bar(tau - 1.5, df["detections"] / 1000, width=3, label="Detections", color="C0", alpha=0.7)
    ax.bar(tau + 1.5, df["spikes"] / 1000, width=3, label="Spikes", color="C3", alpha=0.7)
    ax.set_ylabel("Count (thousands)")
    ax.set_xlabel("Tau (days)")
    ax.set_title("Detections vs Spikes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = BASE_DIR / "tau_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info("Saved plot: %s", plot_path)
    plt.close(fig)

    # Also make a per-date histogram comparison
    make_per_date_plot(df)


def make_per_date_plot(df):
    """Plot detection count per onset date for each tau."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for _, row in df.iterrows():
        tau = int(row["tau"])
        onset_nc = BASE_DIR / f"tau{tau}" / "temporal_onset.nc"
        if not onset_nc.exists():
            continue

        ds = xr.open_dataset(onset_nc)
        cm = ds["candidate_mask"].values
        spike = ds["spike_flag"].values
        valid = cm & ~spike
        onset_dates = ds["onset_date"].values[valid]

        # Get unique dates and counts
        unique_dates, counts = np.unique(onset_dates[~np.isnat(onset_dates)], return_counts=True)
        ax.plot(unique_dates, counts, "o-", label=f"tau={tau}", alpha=0.8, markersize=3)
        ds.close()

    ax.set_xlabel("Onset Date")
    ax.set_ylabel("Detection Count")
    ax.set_title("Detections per Onset Date by Tau")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = BASE_DIR / "tau_per_date_detections.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info("Saved per-date plot: %s", plot_path)
    plt.close(fig)


if __name__ == "__main__":
    df = run_all()
    make_plots(df)
