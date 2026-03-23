"""
Schematic figure showing ANF-based resolution weighting for ISSW paper.

Shows:
- Top row: Synthetic ANF maps for 3 different satellite tracks
- Middle row: Corresponding inverse-resolution weight maps (per-pixel, normalized across tracks)
- Bottom: Bar chart summary of mean weights per track
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(42)

# ── Synthetic ANF data for 3 tracks ────────────────────────────────
ny, nx = 120, 160
y, x = np.mgrid[0:ny, 0:nx]

# Track names (ascending/descending orbit numbers)
track_names = ["Track 54 (Asc)", "Track 129 (Desc)", "Track 32 (Asc)"]

# Create spatially varying ANF fields (meters) with MORE dramatic differences
# ANF depends on local incidence angle which varies with terrain + orbit geometry
# Widen the range to ~15-120 m to make differences more visible

base_terrain = 15 * np.sin(2 * np.pi * x / nx) * np.cos(2 * np.pi * y / ny)

anf_tracks = []
# Track 1: excellent resolution everywhere (~20-35 m) — best track
anf1 = 22 + 6 * (x / nx) + base_terrain * 0.2
anf1 += np.random.normal(0, 1.5, (ny, nx))
anf1 = np.clip(anf1, 15, 120)
anf_tracks.append(anf1)

# Track 2: poor resolution (~60-110 m) — worst track, strong terrain modulation
anf2 = 75 + 25 * np.sin(np.pi * y / ny) + base_terrain * 1.2
anf2 += np.random.normal(0, 3, (ny, nx))
anf2 = np.clip(anf2, 15, 120)
anf_tracks.append(anf2)

# Track 3: mixed — great in upper-left, terrible in lower-right (layover/shadow gradient)
grad = 0.6 * (x / nx) + 0.4 * (y / ny)  # 0 at top-left, ~1 at bottom-right
anf3 = 20 + 80 * grad + base_terrain * 0.5
anf3 += np.random.normal(0, 2.5, (ny, nx))
anf3 = np.clip(anf3, 15, 120)
anf_tracks.append(anf3)

anf_stack = np.stack(anf_tracks, axis=0)  # (3, ny, nx)

# ── Compute inverse-resolution weights ────────────────────────────
inverse_res = 1.0 / anf_stack  # (3, ny, nx)
sum_inv = inverse_res.sum(axis=0, keepdims=True)  # (1, ny, nx)
w_resolution = inverse_res / sum_inv  # (3, ny, nx) — sums to 1 along axis=0

# ── Figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)

anf_norm = Normalize(vmin=15, vmax=110)
w_norm = Normalize(vmin=0.05, vmax=0.70)

# --- Top row: ANF maps ---
for i in range(3):
    ax = axes[0, i]
    im = ax.imshow(anf_tracks[i], cmap="viridis_r", norm=anf_norm, aspect="equal")
    ax.set_title(track_names[i], fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("ANF (m)\n← better resolution", fontsize=11)

divider = make_axes_locatable(axes[0, 2])
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("ANF (m)", fontsize=10)

# --- Middle row: weight maps ---
for i in range(3):
    ax = axes[1, i]
    im2 = ax.imshow(w_resolution[i], cmap="magma", norm=w_norm, aspect="equal")
    mean_w = w_resolution[i].mean()
    ax.text(
        0.03, 0.03,
        f"mean = {mean_w:.2f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("Weight\n← higher weight", fontsize=11)

divider2 = make_axes_locatable(axes[1, 2])
cax2 = divider2.append_axes("right", size="5%", pad=0.08)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.set_label("Weight", fontsize=10)

# --- Bottom row: summary diagram ---
# Merge into one wide axes
for ax in axes[2, :]:
    ax.remove()
ax_summary = fig.add_subplot(3, 1, 3)

# Show combination schematic as stacked bar per track + combined result
mean_anf = [a.mean() for a in anf_tracks]
mean_w = [w_resolution[i].mean() for i in range(3)]

track_colors = ["#1b9e77", "#d95f02", "#7570b3"]
x_pos = np.arange(3)

# Left half: mean ANF bar chart
ax_summary.bar(x_pos - 0.2, mean_anf, 0.35, color=track_colors, alpha=0.7, edgecolor="white")
ax_summary.set_ylabel("Mean ANF (m)", fontsize=12, color="#333333")
ax_summary.set_xticks(x_pos)
ax_summary.set_xticklabels(
    [f"{n}\nANF={a:.0f}m → w={w:.2f}" for n, a, w in zip(track_names, mean_anf, mean_w)],
    fontsize=10,
)
ax_summary.spines[["top", "right"]].set_visible(False)

# Overlay weight on twin axis
ax_w = ax_summary.twinx()
ax_w.bar(x_pos + 0.2, mean_w, 0.35, color=track_colors, alpha=0.35, edgecolor=track_colors, linewidth=1.5, linestyle="--")
ax_w.set_ylabel("Mean weight", fontsize=12, color="#666666")
ax_w.set_ylim(0, 0.75)
ax_w.spines[["top"]].set_visible(False)

ax_summary.set_title(
    r"Inverse resolution weighting:  $w_i = \frac{1/\mathrm{ANF}_i}{\sum_j 1/\mathrm{ANF}_j}$",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

out = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures/resolution_weights_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
