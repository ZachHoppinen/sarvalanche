"""
Schematic figure showing temporal weighting scheme for ISSW paper.

Shows:
- Top: Timeline of SAR acquisitions relative to avalanche date
- Bottom left: Exponential decay curves for tau=6 and tau=24
- Bottom right: Resulting normalized weights for each image under both tau values
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Synthetic data ──────────────────────────────────────────────────
avalanche_date = datetime(2025, 1, 15)

# Simulated SAR acquisition times (days relative to avalanche)
dt_days = np.array([-18, -12, -6, -3, 0, 3, 6, 12])
acq_dates = [avalanche_date + timedelta(days=int(d)) for d in dt_days]
n_images = len(dt_days)

# Continuous curve for plotting
t_cont = np.linspace(-22, 16, 300)

# Compute weights
taus = [6, 24]
colors_tau = ["#2166ac", "#b2182b"]
labels_tau = [r"$\tau = 6$ days", r"$\tau = 24$ days"]


def compute_weights(dt, tau):
    w = np.exp(-np.abs(dt) / tau)
    return w / w.sum()


# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.35)

# --- Top panel: timeline ---
ax_timeline = fig.add_subplot(gs[0, :])
ax_timeline.set_xlim(-23, 17)
ax_timeline.set_ylim(-0.5, 1.5)
ax_timeline.axvline(0, color="red", ls="--", lw=1.5, zorder=2, label="Avalanche date")

for i, d in enumerate(dt_days):
    ax_timeline.annotate(
        "",
        xy=(d, 0.0),
        xytext=(d, 0.8),
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2),
    )
    ax_timeline.text(
        d,
        0.95,
        f"t{i+1}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax_timeline.text(
        d,
        -0.25,
        f"{d:+d}d",
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
    )

ax_timeline.set_xlabel("Days relative to avalanche date", fontsize=11)
ax_timeline.set_yticks([])
ax_timeline.spines[["top", "left", "right"]].set_visible(False)
ax_timeline.legend(loc="upper right", fontsize=10)
ax_timeline.set_title("SAR acquisition timeline", fontsize=12, fontweight="bold")

# --- Bottom left: decay curves ---
ax_decay = fig.add_subplot(gs[1, 0])
for tau, color, label in zip(taus, colors_tau, labels_tau):
    curve = np.exp(-np.abs(t_cont) / tau)
    ax_decay.plot(t_cont, curve, color=color, lw=2, label=label)
    # Mark actual acquisition points
    pts = np.exp(-np.abs(dt_days) / tau)
    ax_decay.scatter(dt_days, pts, color=color, s=40, zorder=5, edgecolors="white", linewidths=0.5)

ax_decay.axvline(0, color="red", ls="--", lw=1, alpha=0.6)
ax_decay.set_xlabel("Days relative to avalanche date", fontsize=11)
ax_decay.set_ylabel(r"Unnormalized weight  $e^{-|\Delta t|/\tau}$", fontsize=11)
ax_decay.set_title("Exponential decay kernel", fontsize=12, fontweight="bold")
ax_decay.legend(fontsize=10)
ax_decay.set_xlim(-23, 17)
ax_decay.set_ylim(-0.02, 1.05)
ax_decay.spines[["top", "right"]].set_visible(False)

# --- Bottom right: normalized bar chart ---
ax_bar = fig.add_subplot(gs[1, 1])
bar_width = 0.35
x_pos = np.arange(n_images)

for j, (tau, color, label) in enumerate(zip(taus, colors_tau, labels_tau)):
    w = compute_weights(dt_days, tau)
    offset = -bar_width / 2 + j * bar_width
    bars = ax_bar.bar(
        x_pos + offset,
        w,
        bar_width,
        color=color,
        alpha=0.85,
        label=label,
        edgecolor="white",
        linewidth=0.5,
    )
    # Annotate bar values
    for bar, val in zip(bars, w):
        if val > 0.04:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
            )

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([f"t{i+1}\n({d:+d}d)" for i, d in enumerate(dt_days)], fontsize=8)
ax_bar.set_ylabel("Normalized weight", fontsize=11)
ax_bar.set_title("Per-image weights (sum = 1)", fontsize=12, fontweight="bold")
ax_bar.legend(fontsize=10)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.15)

# Add equation annotation
fig.text(
    0.5,
    -0.01,
    r"$w_i = \frac{\exp(-|\Delta t_i| / \tau)}{\sum_j \exp(-|\Delta t_j| / \tau)}$",
    ha="center",
    fontsize=13,
)

out = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures/temporal_weights_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
