"""
ISSW figure: spatial amplitude and spatial peak alignment illustration.

For both the high-confidence and low-confidence example pixels, shows:
- Row 1: Raw CNN probability patches at peak time ± a few steps
- Row 2: Gaussian-smoothed patches (what the spatial features measure)
- Row 3: Time series of the pixel vs Gaussian-smoothed neighborhood,
         annotating spatial amplitude and peak alignment.
"""

import warnings
import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────
BASE = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns"
CNN_NC = f"{BASE}/v2_season_inference/season_v2_debris_probabilities.nc"
ONSET_NC = f"{BASE}/v2_season_inference/temporal_onset.nc"
OUT_DIR = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures"

EXAMPLES = {
    "High confidence (real debris)": {"y": 2147, "x": 1145, "color": "#1a9850"},
    "Low confidence (noise spike)":  {"y": 2020, "x": 890,  "color": "#d73027"},
}

THRESHOLD = 0.5
SIGMA = 3.0
SPATIAL_RADIUS = 15  # pixels for patch display
STEPS_AROUND = 3     # time steps before/after peak to show


def gaussian_smooth_frame(frame, sigma):
    valid = ~np.isnan(frame)
    filled = np.where(valid, frame, 0.0)
    bv = ndimage.gaussian_filter(filled, sigma=sigma, mode="constant", cval=0)
    bw = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="constant", cval=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bw > 1e-10, bv / bw, 0.0)


def make_spatial_features_figure():
    print("Loading data...")
    cnn_ds = xr.open_dataset(CNN_NC)
    probs = cnn_ds["debris_probability"].values  # (T, H, W)
    times = pd.DatetimeIndex(cnn_ds["time"].values)
    T, H, W = probs.shape

    onset_ds = xr.open_dataset(ONSET_NC)

    n_examples = len(EXAMPLES)
    n_frames = 2 * STEPS_AROUND + 1

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)

    # Layout: for each example, 3 rows (raw patches, smoothed patches, time series)
    # So 6 rows total, with the patch rows sharing columns
    gs = fig.add_gridspec(
        6, n_frames + 1,
        height_ratios=[1, 1, 1.2, 1, 1, 1.2],
        hspace=0.3,
    )

    prob_norm = Normalize(vmin=0, vmax=0.9)
    prob_cmap = "inferno"

    for ex_idx, (label, pix) in enumerate(EXAMPLES.items()):
        yi, xi = pix["y"], pix["x"]
        accent = pix["color"]
        row_offset = ex_idx * 3  # 0 or 3

        # Get peak index
        pixel_ts = probs[:, yi, xi]
        pk = int(np.nanargmax(pixel_ts))

        # Spatial context metadata
        spat_amp = float(onset_ds["spatial_bump_amplitude"].values[yi, xi])
        spat_align = float(onset_ds["spatial_peak_alignment"].values[yi, xi])

        # Time indices to show
        t_indices = list(range(max(0, pk - STEPS_AROUND), min(T, pk + STEPS_AROUND + 1)))

        # Patch bounds
        r = SPATIAL_RADIUS
        y_lo, y_hi = max(0, yi - r), min(H, yi + r + 1)
        x_lo, x_hi = max(0, xi - r), min(W, xi + r + 1)
        cy_local, cx_local = yi - y_lo, xi - x_lo

        # Precompute Gaussian-smoothed time series at this pixel
        gauss_ts = np.array([gaussian_smooth_frame(probs[t], SIGMA)[yi, xi] for t in range(T)])
        gauss_pk = int(np.argmax(gauss_ts))

        # ── Row A: Raw probability patches ──
        for i, t in enumerate(t_indices):
            ax = fig.add_subplot(gs[row_offset + 0, i])
            patch = probs[t, y_lo:y_hi, x_lo:x_hi]
            ax.imshow(patch, cmap=prob_cmap, norm=prob_norm, aspect="equal")
            ax.plot(cx_local, cy_local, "+", color="white", markersize=10, markeredgewidth=2)

            offset = t - pk
            sign = "+" if offset > 0 else ""
            date_str = times[t].strftime("%m-%d")
            ax.set_title(f"{date_str}\n({sign}{offset})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if t == pk:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)

            if i == 0:
                ax.set_ylabel("Raw P(debris)", fontsize=9)

        # Label + colorbar in last column
        ax_label = fig.add_subplot(gs[row_offset + 0, n_frames])
        ax_label.axis("off")
        ax_label.text(0.1, 0.5, label, fontsize=11, fontweight="bold",
                     color=accent, va="center", rotation=0)

        # ── Row B: Gaussian-smoothed patches ──
        for i, t in enumerate(t_indices):
            ax = fig.add_subplot(gs[row_offset + 1, i])
            smoothed = gaussian_smooth_frame(probs[t], SIGMA)
            patch_smooth = smoothed[y_lo:y_hi, x_lo:x_hi]
            im = ax.imshow(patch_smooth, cmap=prob_cmap, norm=prob_norm, aspect="equal")
            ax.plot(cx_local, cy_local, "+", color="white", markersize=10, markeredgewidth=2)
            ax.set_xticks([])
            ax.set_yticks([])

            if t == pk:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)

            if i == 0:
                ax.set_ylabel(f"Smoothed (σ={SIGMA})", fontsize=9)

        # Colorbar
        ax_cb = fig.add_subplot(gs[row_offset + 1, n_frames])
        plt.colorbar(im, ax=ax_cb, shrink=0.9, label="P(debris)")
        ax_cb.axis("off")

        # ── Row C: Time series with spatial feature annotations ──
        ax_ts = fig.add_subplot(gs[row_offset + 2, :])
        dates = times.to_pydatetime()

        ax_ts.plot(dates, pixel_ts, "o-", markersize=4, linewidth=1.8,
                  label="Pixel probability", color="C0", zorder=3)
        ax_ts.plot(dates, gauss_ts, "s-", markersize=3, linewidth=1.5,
                  label=f"Gaussian neighborhood (σ={SIGMA}px)", color="C1", alpha=0.8)
        ax_ts.axhline(THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)

        # Shade the displayed time window
        ax_ts.axvspan(dates[t_indices[0]], dates[t_indices[-1]],
                     alpha=0.08, color="gray", label="Shown above")

        # Mark pixel peak
        ax_ts.axvline(dates[pk], color="C0", ls="--", lw=1.2, alpha=0.6)
        ax_ts.plot(dates[pk], pixel_ts[pk], "o", color="C0", markersize=10, zorder=5)

        # Mark smoothed peak
        ax_ts.axvline(dates[gauss_pk], color="C1", ls="--", lw=1.2, alpha=0.6)
        ax_ts.plot(dates[gauss_pk], gauss_ts[gauss_pk], "s", color="C1", markersize=10, zorder=5)

        # ── Annotate spatial amplitude ──
        # Baseline = mean of first/last 2 steps of smoothed
        n_edge = min(2, T // 3)
        baseline = (np.mean(gauss_ts[:max(n_edge, 1)]) + np.mean(gauss_ts[-max(n_edge, 1):])) / 2.0

        # Draw amplitude arrow
        amp_x = dates[pk]
        ax_ts.annotate(
            "", xy=(amp_x, gauss_ts[pk]), xytext=(amp_x, baseline),
            arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=2.5),
        )
        ax_ts.text(
            dates[min(pk + 1, T - 1)], (gauss_ts[pk] + baseline) / 2,
            f"  Spatial amplitude\n  = {spat_amp:.2f}",
            fontsize=9, fontweight="bold", color="#e67e22", va="center",
        )
        # Baseline reference line
        ax_ts.axhline(baseline, color="#e67e22", ls=":", lw=1, alpha=0.5)
        ax_ts.text(dates[0], baseline + 0.02, f"baseline = {baseline:.2f}",
                  fontsize=7, color="#e67e22", alpha=0.7)

        # ── Annotate peak alignment ──
        peak_dist = abs(gauss_pk - pk)
        align_score = 1.0 / (1.0 + peak_dist)
        if peak_dist > 0:
            # Draw bracket between the two peaks
            mid_y = max(pixel_ts[pk], gauss_ts[gauss_pk]) + 0.06
            ax_ts.annotate(
                "", xy=(dates[gauss_pk], mid_y), xytext=(dates[pk], mid_y),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2),
            )
            ax_ts.text(
                dates[(pk + gauss_pk) // 2], mid_y + 0.04,
                f"Peak alignment = {align_score:.2f}\n({peak_dist} step{'s' if peak_dist > 1 else ''} apart)",
                fontsize=9, fontweight="bold", color="#8e44ad", ha="center",
            )
        else:
            ax_ts.text(
                dates[min(pk + 2, T - 1)], pixel_ts[pk] + 0.05,
                f"Peak alignment = {align_score:.2f}\n(peaks coincide)",
                fontsize=9, fontweight="bold", color="#8e44ad",
            )

        ax_ts.set_ylim(-0.03, 1.1)
        ax_ts.set_ylabel("Probability", fontsize=10)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax_ts.tick_params(axis="x", rotation=30)
        ax_ts.legend(fontsize=8, loc="upper left", ncol=3)
        ax_ts.grid(True, alpha=0.15)
        ax_ts.spines[["top", "right"]].set_visible(False)
        ax_ts.set_title(label, fontsize=11, fontweight="bold", color=accent)

        if ex_idx == 1:
            ax_ts.set_xlabel("Date (2024-2025 season)", fontsize=10)

    fig.suptitle(
        "Spatial Features: Amplitude & Peak Alignment",
        fontsize=15, fontweight="bold",
    )

    out = f"{OUT_DIR}/spatial_features_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    make_spatial_features_figure()
