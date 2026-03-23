"""
ISSW figure: d_empirical (backscatter change) vs CNN debris probability.

Shows side-by-side comparisons for a single date (2025-02-04) at multiple
zoom levels: full scene overview + 2 zoomed-in avalanche clusters.
"""

import logging
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.detection.backscatter_change import (
    calculate_empirical_backscatter_probability,
)
from sarvalanche.weights.temporal import get_temporal_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────
NC_PATH = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
CNN_PATH = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference/scene_v2_debris_2025-02-04.tif"
REF_DATE = pd.Timestamp("2025-02-04")
TAU = 6

# Zoom boxes: (y_slice, x_slice, label)
ZOOMS = [
    (slice(2000, 2200), slice(850, 1150), "Cluster A: Galena Summit"),
    (slice(1580, 1750), slice(30, 250), "Cluster B: Western Smokys"),
]

# ── Load data ──────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_netcdf_to_dataset(NC_PATH)
if any(var.chunks is not None for var in ds.variables.values()):
    ds = ds.load()

print("Loading CNN output...")
cnn_full = xr.open_dataarray(CNN_PATH).squeeze(drop=True)

# ── Compute d_empirical ────────────────────────────────────────────
print("Computing d_empirical...")
ds["w_temporal"] = get_temporal_weights(ds["time"], np.datetime64(REF_DATE), tau_days=TAU)
ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

from sarvalanche.weights.local_resolution import get_local_resolution_weights

if "w_resolution" not in ds.data_vars:
    ds["w_resolution"] = get_local_resolution_weights(ds["anf"])

p_emp, d_emp = calculate_empirical_backscatter_probability(
    ds, np.datetime64(REF_DATE),
    use_agreement_boosting=True,
    agreement_strength=0.8,
    min_prob_threshold=0.2,
    tau_days=TAU,
)

d_emp_vals = d_emp.values
cnn_vals = cnn_full.values

# Also get hillshade from DEM for context
dem = ds["dem"].values
# Simple hillshade
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(np.nan_to_num(dem, nan=0), vert_exag=2)

print(f"d_empirical range: [{np.nanmin(d_emp_vals):.2f}, {np.nanmax(d_emp_vals):.2f}] dB")
print(f"CNN range: [{np.nanmin(cnn_vals):.4f}, {np.nanmax(cnn_vals):.4f}]")

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 16), constrained_layout=True)

# Layout: 3 rows x 2 cols
# Row 0: full scene overview (d_emp | CNN)
# Row 1: zoom A (d_emp | CNN)
# Row 2: zoom B (d_emp | CNN)
gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1], hspace=0.15, wspace=0.08)

d_norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3.0)
cnn_norm = Normalize(vmin=0, vmax=0.8)

zoom_colors = ["#e41a1c", "#377eb8"]

# ── Row 0: Full scene ──────────────────────────────────────────────
ax_d_full = fig.add_subplot(gs[0, 0])
ax_d_full.imshow(hillshade, cmap="gray", alpha=0.3, aspect="equal")
im_d = ax_d_full.imshow(
    d_emp_vals, cmap="RdBu_r", norm=d_norm, alpha=0.85, aspect="equal",
)
ax_d_full.set_title(f"d_empirical (τ={TAU}d)", fontsize=13, fontweight="bold")
ax_d_full.set_xticks([])
ax_d_full.set_yticks([])
plt.colorbar(im_d, ax=ax_d_full, shrink=0.7, label="Backscatter change (dB)")

# Draw zoom boxes
for i, (ys, xs, label) in enumerate(ZOOMS):
    rect = Rectangle(
        (xs.start, ys.start),
        xs.stop - xs.start,
        ys.stop - ys.start,
        linewidth=2, edgecolor=zoom_colors[i], facecolor="none", linestyle="--",
    )
    ax_d_full.add_patch(rect)
    ax_d_full.text(
        xs.start, ys.start - 15, label,
        color=zoom_colors[i], fontsize=9, fontweight="bold",
    )

ax_c_full = fig.add_subplot(gs[0, 1])
ax_c_full.imshow(hillshade, cmap="gray", alpha=0.3, aspect="equal")
im_c = ax_c_full.imshow(
    cnn_vals, cmap="inferno", norm=cnn_norm, alpha=0.85, aspect="equal",
)
ax_c_full.set_title("CNN debris probability", fontsize=13, fontweight="bold")
ax_c_full.set_xticks([])
ax_c_full.set_yticks([])
plt.colorbar(im_c, ax=ax_c_full, shrink=0.7, label="P(debris)")

# Draw same zoom boxes on CNN
for i, (ys, xs, label) in enumerate(ZOOMS):
    rect = Rectangle(
        (xs.start, ys.start),
        xs.stop - xs.start,
        ys.stop - ys.start,
        linewidth=2, edgecolor=zoom_colors[i], facecolor="none", linestyle="--",
    )
    ax_c_full.add_patch(rect)

# ── Zoom rows ──────────────────────────────────────────────────────
for row, (ys, xs, label) in enumerate(ZOOMS, start=1):
    color = zoom_colors[row - 1]

    # d_empirical zoom
    ax_dz = fig.add_subplot(gs[row, 0])
    ax_dz.imshow(hillshade[ys, xs], cmap="gray", alpha=0.3, aspect="equal")
    im_dz = ax_dz.imshow(
        d_emp_vals[ys, xs], cmap="RdBu_r", norm=d_norm, alpha=0.85, aspect="equal",
    )
    ax_dz.set_title(f"{label} — d_empirical", fontsize=11, fontweight="bold", color=color)
    ax_dz.set_xticks([])
    ax_dz.set_yticks([])
    # Border matching zoom box color
    for spine in ax_dz.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    plt.colorbar(im_dz, ax=ax_dz, shrink=0.8, label="dB")

    # CNN zoom
    ax_cz = fig.add_subplot(gs[row, 1])
    ax_cz.imshow(hillshade[ys, xs], cmap="gray", alpha=0.3, aspect="equal")
    im_cz = ax_cz.imshow(
        cnn_vals[ys, xs], cmap="inferno", norm=cnn_norm, alpha=0.85, aspect="equal",
    )
    ax_cz.set_title(f"{label} — CNN probability", fontsize=11, fontweight="bold", color=color)
    ax_cz.set_xticks([])
    ax_cz.set_yticks([])
    for spine in ax_cz.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    plt.colorbar(im_cz, ax=ax_cz, shrink=0.8, label="P(debris)")

fig.suptitle(
    f"2025-02-04 — Sawtooth & Western Smoky Mtns",
    fontsize=15, fontweight="bold", y=1.01,
)

out = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures/cnn_inference_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
