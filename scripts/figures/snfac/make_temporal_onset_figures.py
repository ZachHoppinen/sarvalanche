"""
ISSW figures for temporal onset / confidence classification.

Figure 1: Example time series — high-confidence real debris vs low-confidence noise,
          showing bump width, persistence, spatial alignment, etc.

Figure 2: Workflow schematic — how temporal features are extracted and combined
          into a confidence score.
"""

import warnings
import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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

# Example pixels (from data exploration)
EXAMPLES = {
    "High confidence\n(real debris)": {"y": 2147, "x": 1145},
    "Low confidence\n(noise spike)":  {"y": 2020, "x": 890},
}

THRESHOLD = 0.5
SIGMA = 3.0  # Gaussian spatial smoothing


def gaussian_smooth_frame(frame, sigma):
    valid = ~np.isnan(frame)
    filled = np.where(valid, frame, 0.0)
    bv = ndimage.gaussian_filter(filled, sigma=sigma, mode="constant", cval=0)
    bw = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="constant", cval=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bw > 1e-10, bv / bw, 0.0)


# =====================================================================
# FIGURE 1: Time series examples
# =====================================================================

def make_timeseries_figure():
    print("Loading data for time series figure...")
    cnn_ds = xr.open_dataset(CNN_NC)
    probs = cnn_ds["debris_probability"].values  # (T, H, W)
    times = pd.DatetimeIndex(cnn_ds["time"].values)
    T = len(times)

    onset_ds = xr.open_dataset(ONSET_NC)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2.5, 1]})

    for row, (label, pix) in enumerate(EXAMPLES.items()):
        yi, xi = pix["y"], pix["x"]

        # Extract time series
        pixel_ts = probs[:, yi, xi]
        pk = int(np.nanargmax(pixel_ts))

        # Gaussian-smoothed neighborhood
        gauss_ts = np.array([gaussian_smooth_frame(probs[t], SIGMA)[yi, xi] for t in range(T)])

        # Onset metadata
        conf = float(onset_ds["confidence"].values[yi, xi])
        bw = int(onset_ds["bump_width"].values[yi, xi])
        spike = bool(onset_ds["spike_flag"].values[yi, xi])
        spat_amp = float(onset_ds["spatial_bump_amplitude"].values[yi, xi])
        spat_align = float(onset_ds["spatial_peak_alignment"].values[yi, xi])
        smooth = float(onset_ds["bump_smoothness"].values[yi, xi])

        # -- Left panel: time series --
        ax = axes[row, 0]
        dates = times.to_pydatetime()

        ax.fill_between(dates, 0, pixel_ts, alpha=0.15, color="C0")
        ax.plot(dates, pixel_ts, "o-", markersize=4, linewidth=1.8, label="Pixel probability", color="C0", zorder=3)
        ax.plot(dates, gauss_ts, "s-", markersize=3, linewidth=1.5, label=f"Spatial neighborhood (σ={SIGMA}px)", color="C1", alpha=0.8)

        # Threshold line
        ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Threshold = {THRESHOLD}")

        # Shade bump region
        above = pixel_ts >= THRESHOLD
        left = pk
        while left > 0 and above[left - 1]:
            left -= 1
        right = pk
        while right < T - 1 and above[right + 1]:
            right += 1

        if above[pk]:
            ax.axvspan(dates[left], dates[right], alpha=0.12, color="green", label=f"Bump width = {bw} steps")

        # Mark peak
        ax.axvline(dates[pk], color="red", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.annotate(
            f"Peak: {dates[pk].strftime('%Y-%m-%d')}\nP = {pixel_ts[pk]:.2f}",
            xy=(dates[pk], pixel_ts[pk]),
            xytext=(15, -10), textcoords="offset points",
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )

        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("CNN probability", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold",
                     color="#1a9850" if "High" in label else "#d73027")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

        if row == 1:
            ax.set_xlabel("Date (2024-2025 season)", fontsize=11)

        # -- Right panel: feature summary --
        ax_feat = axes[row, 1]
        ax_feat.axis("off")

        features = [
            ("Confidence", f"{conf:.2f}", conf),
            ("Bump width", f"{bw} steps", min(bw / 8, 1)),
            ("Spatial amplitude", f"{spat_amp:.2f}", min(spat_amp / 0.5, 1)),
            ("Spatial alignment", f"{spat_align:.2f}", spat_align),
            ("Bump smoothness", f"{smooth:.2f}", smooth),
            ("Spike flag", "Yes" if spike else "No", 0.0 if spike else 1.0),
        ]

        y_start = 0.92
        for i, (fname, fval, score) in enumerate(features):
            y_pos = y_start - i * 0.13

            # Color bar showing score
            bar_color = plt.cm.RdYlGn(score)
            ax_feat.barh(y_pos, score, height=0.08, left=0.45, color=bar_color,
                        edgecolor="gray", linewidth=0.5, transform=ax_feat.transAxes)
            ax_feat.barh(y_pos, 1.0, height=0.08, left=0.45, color="none",
                        edgecolor="gray", linewidth=0.5, transform=ax_feat.transAxes)

            ax_feat.text(0.42, y_pos, fname, ha="right", va="center",
                        fontsize=9, transform=ax_feat.transAxes)
            ax_feat.text(0.45 + score + 0.03, y_pos, fval, ha="left", va="center",
                        fontsize=9, fontweight="bold", transform=ax_feat.transAxes,
                        color=plt.cm.RdYlGn(score))

        # Confidence box
        conf_color = "#1a9850" if conf > 0.5 else "#d73027"
        ax_feat.text(
            0.7, 0.05, f"Confidence: {conf:.2f}",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", transform=ax_feat.transAxes,
            bbox=dict(facecolor=conf_color, edgecolor="none", boxstyle="round,pad=0.4"),
        )

    plt.tight_layout()
    out = f"{OUT_DIR}/temporal_onset_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


# =====================================================================
# FIGURE 2: Workflow schematic
# =====================================================================

def make_workflow_figure():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", edgecolor="#333333", linewidth=1.5)
    title_fs = 11
    body_fs = 8.5

    def draw_box(x, y, w, h, title, body, facecolor="#e8f4f8", title_color="#333"):
        rect = FancyBboxPatch((x, y), w, h, **box_kw, facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=title_color)
        ax.text(x + w/2, y + h/2 - 0.15, body, ha="center", va="center",
                fontsize=body_fs, color="#444", linespacing=1.5)

    def draw_arrow(x1, y1, x2, y2, color="#666"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

    # ── Title ──
    ax.text(7, 8.7, "Temporal Onset & Confidence Pipeline", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(7, 8.35, "Stage 2: When did the avalanche occur? How confident are we?",
            ha="center", fontsize=11, color="#666", style="italic")

    # ── Input box ──
    draw_box(0.3, 6.8, 3.2, 1.2, "Input: CNN Probability Cube",
             "debris_probability(time, y, x)\n47 dates × 2945 × 1907",
             facecolor="#d4e6f1")

    # ── Peak finding ──
    draw_box(5.0, 6.8, 3.8, 1.2, "1. Peak Detection",
             "For each pixel, find time of\nmax P(debris) above threshold",
             facecolor="#fdebd0")

    draw_arrow(3.5, 7.4, 5.0, 7.4)

    # ── Temporal features ──
    draw_box(0.5, 4.5, 5.5, 1.9, "2. Temporal Features",
             "• Bump width — contiguous steps > threshold\n"
             "• N above threshold — total confirming passes\n"
             "• Mean detection probability\n"
             "• Persistence ratio — fraction of span > 0.3\n"
             "• Bump smoothness — low 2nd derivative",
             facecolor="#d5f5e3")

    draw_arrow(5.0, 6.8, 3.25, 6.4)

    # ── Spatial features ──
    draw_box(7.5, 4.5, 5.5, 1.9, "3. Spatial Context",
             "Gaussian smooth prob cube (σ=3 px):\n"
             "• Bump amplitude — peak vs baseline\n"
             "• Peak alignment — neighborhood peaks\n"
             "  at same time as pixel?\n"
             "• Bump symmetry — rise ≈ fall?",
             facecolor="#e8daef")

    draw_arrow(8.8, 6.8, 10.25, 6.4)

    # ── Confidence score ──
    draw_box(3.0, 2.2, 8.0, 1.9, "4. Confidence Score (0 → 1)",
             "Weighted combination:\n"
             "  25% bump_width  +  20% spatial_alignment  +  20% spatial_amplitude\n"
             "  10% n_above  +  10% mean_det_prob  +  10% persistence\n"
             "  5% smoothness",
             facecolor="#fadbd8")

    draw_arrow(3.25, 4.5, 5.5, 4.1)
    draw_arrow(10.25, 4.5, 8.5, 4.1)

    # ── Spike filter ──
    draw_box(0.3, 2.2, 2.8, 1.2, "Spike Filter",
             "bump_width < 2 →\nflag as noise\nconf × 0.3",
             facecolor="#f9e79f")

    draw_arrow(2.4, 4.5, 1.7, 3.4)

    # ── Outputs ──
    draw_box(3.5, 0.2, 3.3, 1.6, "High Confidence",
             "• Width ≥ 3 steps\n• Spatially coherent\n• Persistent signal\n→ Real avalanche debris",
             facecolor="#82e0aa", title_color="#1a6e33")

    draw_box(7.2, 0.2, 3.3, 1.6, "Low Confidence",
             "• Width = 1 (spike)\n• No spatial coherence\n• Transient signal\n→ Likely noise",
             facecolor="#f1948a", title_color="#922b21")

    draw_arrow(5.2, 2.2, 5.2, 1.8)
    draw_arrow(8.8, 2.2, 8.8, 1.8)

    # ── Key insight callout ──
    ax.text(11.8, 2.7, "Key insight:", fontsize=10, fontweight="bold", color="#2c3e50",
            ha="left")
    ax.text(11.8, 2.1,
            "Each above-threshold\ntime step ≈ independent\n"
            "SAR pass confirming\nthe detection.\n\n"
            "More passes = higher\nconfidence in real debris.",
            fontsize=8.5, color="#555", ha="left", va="top",
            bbox=dict(facecolor="#fef9e7", edgecolor="#d4ac0d", boxstyle="round,pad=0.4", linewidth=1.2))

    out = f"{OUT_DIR}/temporal_onset_workflow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    make_timeseries_figure()
    make_workflow_figure()
