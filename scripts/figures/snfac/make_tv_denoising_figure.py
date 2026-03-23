"""
Schematic figure showing homomorphic TV despeckling for ISSW paper.

Shows:
- Row 1: Original speckled SAR image (linear) → log transform → TV denoise → exp back
- Row 2: Before/after comparison in dB with difference map
- Bottom: 1D cross-section profile showing speckle reduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.restoration import denoise_tv_chambolle

np.random.seed(7)

# ── Synthetic SAR-like image (linear gamma0) ───────────────────────
ny, nx = 200, 200
y, x = np.mgrid[0:ny, 0:nx]

# Underlying "true" backscatter scene in dB (smooth features)
# Simulate a mountain scene: bright slopes, dark shadows, some avalanche paths
scene_db = -15 * np.ones((ny, nx))

# Ridge (bright)
ridge = np.exp(-((y - 60) ** 2) / (2 * 20**2))
scene_db += 6 * ridge

# Valley shadow (dark)
shadow = np.exp(-((y - 150) ** 2 + (x - 120) ** 2) / (2 * 30**2))
scene_db -= 5 * shadow

# Avalanche path (distinct backscatter change)
avy_path = np.exp(-((x - 80) ** 2) / (2 * 12**2)) * np.exp(-((y - 110) ** 2) / (2 * 40**2))
scene_db += 4 * avy_path

# Smooth gradient across scene
scene_db += 3 * (x / nx) - 1.5

# Convert to linear
scene_linear = 10 ** (scene_db / 10)

# Add multiplicative speckle noise (Rayleigh-distributed for single-look SAR)
# For multi-look, approximate as gamma distribution — use ~4 looks
n_looks = 4
speckle = np.random.gamma(n_looks, 1.0 / n_looks, (ny, nx))
noisy_linear = scene_linear * speckle

# ── Homomorphic TV denoising (matches our pipeline) ───────────────
# Step 1: linear → dB
noisy_db = 10 * np.log10(np.clip(noisy_linear, 1e-10, None))

# Step 2: TV denoise in dB domain
tv_weight = 2.0  # stronger denoising for illustration
denoised_db = denoise_tv_chambolle(noisy_db, weight=tv_weight)

# Step 3: dB → linear
denoised_linear = 10 ** (denoised_db / 10)

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11), constrained_layout=True)

# Layout: 2 rows of images + 1 row cross-section
gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1.0], hspace=0.3)

db_norm = Normalize(vmin=-22, vmax=-6)
diff_norm = Normalize(vmin=-3, vmax=3)

# ── Row 1: Pipeline flow (4 steps) ────────────────────────────────
titles_row1 = [
    "1. Original (linear γ⁰)",
    "2. Convert to dB",
    "3. TV denoise (dB domain)",
    "4. Convert back to linear",
]
images_row1 = [noisy_linear, noisy_db, denoised_db, denoised_linear]
cmaps_row1 = ["gray", "gray", "gray", "gray"]
norms_row1 = [
    Normalize(vmin=0, vmax=np.percentile(noisy_linear, 98)),
    db_norm,
    db_norm,
    Normalize(vmin=0, vmax=np.percentile(noisy_linear, 98)),
]

for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(images_row1[i], cmap=cmaps_row1[i], norm=norms_row1[i], aspect="equal")
    ax.set_title(titles_row1[i], fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    # Add arrow between panels
    if i < 3:
        ax.annotate(
            "",
            xy=(1.08, 0.5),
            xycoords="axes fraction",
            xytext=(1.02, 0.5),
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=2),
        )

# ── Row 2: Before / After / Difference (dB domain) ───────────────
ax_before = fig.add_subplot(gs[1, 0:2])
im_b = ax_before.imshow(noisy_db, cmap="gray", norm=db_norm, aspect="equal")
ax_before.set_title("Before: noisy (dB)", fontsize=11, fontweight="bold")
ax_before.set_xticks([])
ax_before.set_yticks([])
plt.colorbar(im_b, ax=ax_before, shrink=0.8, label="dB")

# Draw the cross-section line
row_idx = 110
ax_before.axhline(row_idx, color="#cc0000", ls="--", lw=1.2, alpha=0.8)
ax_before.text(nx - 2, row_idx - 5, "cross-section", ha="right", fontsize=8, color="#cc0000")

ax_after = fig.add_subplot(gs[1, 2])
im_a = ax_after.imshow(denoised_db, cmap="gray", norm=db_norm, aspect="equal")
ax_after.set_title("After: TV denoised (dB)", fontsize=11, fontweight="bold")
ax_after.set_xticks([])
ax_after.set_yticks([])
ax_after.axhline(row_idx, color="#cc0000", ls="--", lw=1.2, alpha=0.8)
plt.colorbar(im_a, ax=ax_after, shrink=0.8, label="dB")

ax_diff = fig.add_subplot(gs[1, 3])
diff = noisy_db - denoised_db
im_d = ax_diff.imshow(diff, cmap="RdBu_r", norm=diff_norm, aspect="equal")
ax_diff.set_title("Removed speckle", fontsize=11, fontweight="bold")
ax_diff.set_xticks([])
ax_diff.set_yticks([])
plt.colorbar(im_d, ax=ax_diff, shrink=0.8, label="ΔdB")

# Stats annotation
std_before = np.nanstd(noisy_db)
std_after = np.nanstd(denoised_db)
reduction = (1 - std_after / std_before) * 100
ax_diff.text(
    0.5, -0.08,
    f"Std reduction: {reduction:.0f}%",
    transform=ax_diff.transAxes,
    ha="center", fontsize=10, color="#333",
)

# ── Row 3: Cross-section profiles ─────────────────────────────────
ax_profile = fig.add_subplot(gs[2, :])

xs = np.arange(nx)
ax_profile.plot(xs, noisy_db[row_idx, :], color="#888888", lw=0.8, alpha=0.7, label="Noisy (dB)")
ax_profile.plot(xs, denoised_db[row_idx, :], color="#cc0000", lw=2, label="TV denoised (dB)")
ax_profile.plot(xs, scene_db[row_idx, :], color="#2166ac", lw=1.5, ls="--", label="True scene (dB)")

ax_profile.set_xlabel("Pixel (x)", fontsize=11)
ax_profile.set_ylabel("Backscatter (dB)", fontsize=11)
ax_profile.set_title(
    f"Cross-section at y={row_idx}  —  TV weight = {tv_weight}",
    fontsize=11,
    fontweight="bold",
)
ax_profile.legend(fontsize=10, loc="lower right")
ax_profile.spines[["top", "right"]].set_visible(False)
ax_profile.set_xlim(0, nx - 1)

# Add formula annotation
fig.text(
    0.5, -0.01,
    r"Homomorphic TV:   $\hat{x}_{\mathrm{dB}} = \arg\min_x \left\{ \|x - y_{\mathrm{dB}}\|_2^2 + \lambda \, \|\nabla x\|_1 \right\}$"
    f"     (λ = {tv_weight})",
    ha="center",
    fontsize=12,
)

out = "/Users/zmhoppinen/Documents/sarvalanche/local/issw/figures/tv_denoising_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
