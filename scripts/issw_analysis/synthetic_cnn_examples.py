#!/usr/bin/env python3
"""Generate CNN sensitivity figures for ISSW using real training patches.

Strategy: take a real positive patch where the CNN gives strong output
(prob ~0.95 in debris), then systematically modify one input at a time
to show how terrain and SAR features affect the detection.

This avoids the domain-gap problem of fully synthetic patches —
the CNN was trained on real SAR textures and responds weakly to smooth
synthetic fields.

Saves figures to local/issw/figures/cnn_synthetic/
"""

import glob
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarvalanche.ml.v2.channels import STATIC_CHANNELS, N_STATIC, normalize_static_channel
from sarvalanche.ml.v2.model import DebrisDetector, DebrisDetectorSkip
from sarvalanche.ml.v2.patch_extraction import DEM_CHANNEL, normalize_dem_patch

WEIGHTS = ROOT / "local/issw/v2_patches/v2_detector_best.pt"
PATCH_DIR = ROOT / "local/issw/v2_patches/2024-11-15"
OUT_DIR = ROOT / "local/issw/figures/cnn_synthetic"

CH = {name: i for i, name in enumerate(STATIC_CHANNELS)}
PATCH = 128


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_model():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    # Detect architecture from state_dict keys
    key0 = next(iter(state.keys()))
    if "block1" in key0 or any("skip" in k for k in state.keys()):
        log.info("Detected DebrisDetectorSkip (skip-connection architecture)")
        model = DebrisDetectorSkip()
    else:
        log.info("Detected DebrisDetector (no skip connections)")
        model = DebrisDetector()
    model.load_state_dict(state)
    model.eval()
    return model


def run_cnn(model, sar_maps_np, static_stack, device="cpu"):
    """Run CNN. sar_maps_np: (N, 2, 128, 128), static_stack: (10, 128, 128)."""
    static_normed = normalize_dem_patch(static_stack.copy())
    sar_batch = [
        torch.from_numpy(sar_maps_np[i:i+1]).float().to(device)
        for i in range(sar_maps_np.shape[0])
    ]
    static_t = torch.from_numpy(static_normed[np.newaxis]).float().to(device)
    with torch.no_grad():
        logits = model(sar_batch, static_t)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return prob


def load_best_positive_patch():
    """Load the positive patch with highest debris fraction for clear visualization."""
    pos_files = sorted(glob.glob(str(PATCH_DIR / "pos_*.npz")))
    best = None
    best_frac = 0
    for pf in pos_files:
        p = np.load(pf)
        frac = p["label_mask"].mean()
        if frac > best_frac:
            best_frac = frac
            best = pf
    log.info("Best patch: %s (debris frac=%.3f)", Path(best).name, best_frac)
    return np.load(best)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_4panel(
    sar_ch0, static, prob, debris_mask,
    title, subtitle, out_path,
    extra_label=None,
):
    """4-panel: SAR change | terrain features | CNN output | prob in context."""
    fig = plt.figure(figsize=(22, 5.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.28)

    # Panel 1: best SAR track
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(sar_ch0, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                     origin="upper", interpolation="nearest")
    if debris_mask is not None:
        ax1.contour(debris_mask, levels=[0.5], colors="lime", linewidths=1.2,
                    linestyles="--")
    fig.colorbar(im1, ax=ax1, shrink=0.7, label="log1p change")
    ax1.set_title("SAR backscatter change", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: slope
    slope_norm = static[CH["slope"]]
    slope_deg = slope_norm * 0.6 * (180 / np.pi)  # undo norm, convert to degrees
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(slope_deg, cmap="YlOrRd", origin="upper", vmin=0, vmax=55)
    if debris_mask is not None:
        ax2.contour(debris_mask, levels=[0.5], colors="cyan", linewidths=1.2,
                    linestyles="--")
    fig.colorbar(im2, ax=ax2, shrink=0.7, label="Slope (°)")
    ax2.set_title("Slope", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: cell_counts + other features
    ax3 = fig.add_subplot(gs[2])
    cc_raw = static[CH["cell_counts"]]
    cc_display = np.sign(cc_raw) * (np.exp(np.abs(cc_raw) * 5.0) - 1)
    im3 = ax3.imshow(cc_display, cmap="YlOrRd", origin="upper",
                     vmin=0, vmax=max(cc_display.max(), 1))
    fig.colorbar(im3, ax=ax3, shrink=0.7, label="cell_counts")
    rz = static[CH["release_zones"]]
    if rz.max() > 0:
        ax3.contour(rz, levels=[0.5], colors="lime", linewidths=1)
    wm = static[CH["water_mask"]]
    if wm.max() > 0:
        ax3.contourf(wm, levels=[0.5, 1.5], colors=["blue"], alpha=0.3)
    fcf = static[CH["fcf"]]
    if fcf.max() > 0.1:
        ax3.contourf(fcf * 100, levels=[10, 40, 70, 100],
                     colors=["#88cc88", "#44aa44", "#227722"],
                     alpha=0.3, origin="upper")
    if debris_mask is not None:
        ax3.contour(debris_mask, levels=[0.5], colors="cyan", linewidths=1,
                    linestyles="--")
    ax3.set_title("Terrain features", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: CNN probability
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(prob, cmap="hot_r", vmin=0, vmax=1, origin="upper",
                     interpolation="nearest")
    if debris_mask is not None:
        ax4.contour(debris_mask, levels=[0.5], colors="cyan", linewidths=1.2,
                    linestyles="--")
    fig.colorbar(im4, ax=ax4, shrink=0.7, label="P(debris)")
    ax4.set_title(f"CNN output (max={prob.max():.3f})", fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])
    if extra_label:
        ax4.text(0.02, 0.02, extra_label, transform=ax4.transAxes,
                 fontsize=8, va="bottom", color="white",
                 bbox=dict(facecolor="black", alpha=0.7, pad=2))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    if subtitle:
        fig.text(0.5, 0.97, subtitle, ha="center", fontsize=10,
                 style="italic", color="gray")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


def plot_comparison_grid(scenarios, out_path, suptitle):
    """Grid: top=key input, bottom=CNN prob. Each scenario is a tuple:
    (label, display_2d, prob, debris_mask, stats_str)
    """
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (label, display, prob, debris_mask, stats) in enumerate(scenarios):
        # Top: the input being varied
        if display.ndim == 2:
            im1 = axes[0, i].imshow(display, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                                    origin="upper", interpolation="nearest")
        else:
            im1 = axes[0, i].imshow(display, cmap="YlOrRd", origin="upper")
        if debris_mask is not None:
            axes[0, i].contour(debris_mask, levels=[0.5], colors="lime",
                               linewidths=0.8, linestyles="--")
        axes[0, i].set_title(label, fontsize=10, fontweight="bold")
        if i == 0:
            axes[0, i].set_ylabel("Modified input", fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        fig.colorbar(im1, ax=axes[0, i], shrink=0.7)

        # Bottom: CNN output
        im2 = axes[1, i].imshow(prob, cmap="hot_r", vmin=0, vmax=1,
                                origin="upper", interpolation="nearest")
        if debris_mask is not None:
            axes[1, i].contour(debris_mask, levels=[0.5], colors="cyan",
                               linewidths=0.8, linestyles="--")
        axes[1, i].set_title(f"max P = {prob.max():.3f}", fontsize=10)
        if i == 0:
            axes[1, i].set_ylabel("CNN P(debris)", fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        fig.colorbar(im2, ax=axes[1, i], shrink=0.7)

        if stats:
            axes[1, i].text(
                0.02, 0.02, stats, transform=axes[1, i].transAxes,
                fontsize=7, va="bottom", color="white",
                bbox=dict(facecolor="black", alpha=0.7, pad=2),
            )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


def find_best_sar_track(sar_maps, debris_mask):
    """Return index of the SAR track with strongest debris-vs-bg contrast."""
    best_idx = 0
    best_diff = -999
    for i in range(sar_maps.shape[0]):
        ch0 = sar_maps[i, 0]
        if ch0.std() < 0.01:
            continue
        debris = debris_mask > 0.5
        diff = ch0[debris].mean() - ch0[~debris].mean()
        if diff > best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    log.info("Model loaded")

    # Load real patch
    patch = load_best_positive_patch()
    sar_orig = patch["sar_maps"].copy()       # (N, 2, 128, 128)
    static_orig = patch["static"].copy()      # (10, 128, 128)
    mask = patch["label_mask"]                 # (128, 128)
    debris = mask > 0.5

    best_track = find_best_sar_track(sar_orig, mask)
    sar_display = sar_orig[best_track, 0]  # best track change channel for display
    log.info("Best SAR track: %d (debris mean=%.2f, bg mean=%.2f)",
             best_track, sar_display[debris].mean(), sar_display[~debris].mean())

    # Baseline probability
    prob_base = run_cnn(model, sar_orig, static_orig)
    log.info("Baseline: max_prob=%.3f, prob_in_debris=%.3f",
             prob_base.max(), prob_base[debris].mean())

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 1: Baseline — real patch, unmodified
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 1: Baseline")
    plot_4panel(
        sar_display, static_orig, prob_base, mask,
        "Scenario 1: Real training patch — CNN baseline",
        f"Unmodified positive patch | {sar_orig.shape[0]} tracks, "
        f"debris={mask.mean():.1%} of patch",
        OUT_DIR / "01_baseline_real.png",
        extra_label=f"prob in debris={prob_base[debris].mean():.3f}\nprob in bg={prob_base[~debris].mean():.3f}",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 2: Flatten the slope to ~21° (like D3.0 miss)
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 2: Slope override")
    slope_configs = [
        ("Original slope", None),
        ("Slope = 35°\n(steep track)", np.radians(35)),
        ("Slope = 21°\n(D3.0 miss)", np.radians(21)),
        ("Slope = 10°\n(flat valley)", np.radians(10)),
    ]
    slope_scenarios = []
    for label, override_rad in slope_configs:
        s = static_orig.copy()
        if override_rad is not None:
            s[CH["slope"]] = normalize_static_channel(
                np.full((PATCH, PATCH), override_rad, dtype=np.float32), "slope"
            )
        p = run_cnn(model, sar_orig, s)
        slope_deg_display = s[CH["slope"]] * 0.6 * (180 / np.pi)
        slope_scenarios.append(
            (label, slope_deg_display, p, mask,
             f"mean slope={slope_deg_display.mean():.0f}°\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        slope_scenarios,
        OUT_DIR / "02_slope_sensitivity.png",
        "Slope Sensitivity — same SAR signal, overriding slope channel",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 3: Cell counts override
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 3: Cell counts")
    cc_configs = [
        ("Original", None),
        ("cell_counts = 200", 200),
        ("cell_counts = 8\n(D3.0 miss)", 8),
        ("cell_counts = 0\n(no path)", 0),
    ]
    cc_scenarios = []
    for label, cc_val in cc_configs:
        s = static_orig.copy()
        if cc_val is not None:
            s[CH["cell_counts"]] = normalize_static_channel(
                np.full((PATCH, PATCH), float(cc_val), dtype=np.float32), "cell_counts"
            )
        p = run_cnn(model, sar_orig, s)
        cc_display = np.sign(s[CH["cell_counts"]]) * (
            np.exp(np.abs(s[CH["cell_counts"]]) * 5.0) - 1
        )
        cc_scenarios.append(
            (label, cc_display, p, mask,
             f"prob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        cc_scenarios,
        OUT_DIR / "03_cell_counts_sensitivity.png",
        "Cell Counts Sensitivity — same SAR signal, overriding FlowPy convergence",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 4: Water mask
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 4: Water mask")
    water_configs = [
        ("No water\n(original)", 0.0),
        ("5% water", 0.05),
        ("15% water\n(D3.0 miss)", 0.15),
        ("40% water", 0.40),
    ]
    water_scenarios = []
    rng = np.random.RandomState(42)
    for label, frac in water_configs:
        s = static_orig.copy()
        sar_mod = sar_orig.copy()
        wm = np.zeros((PATCH, PATCH), dtype=np.float32)
        if frac > 0:
            wm_pix = rng.random((PATCH, PATCH)) < frac
            wm[wm_pix] = 1.0
            # Water corrupts SAR signal
            for t in range(sar_mod.shape[0]):
                noise = rng.normal(0, 1.5, sar_mod[t, 0][wm > 0.5].shape).astype(np.float32)
                sar_mod[t, 0][wm > 0.5] = noise
        s[CH["water_mask"]] = wm
        p = run_cnn(model, sar_mod, s)
        water_scenarios.append(
            (label, sar_mod[best_track, 0], p, mask,
             f"water={frac:.0%}\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        water_scenarios,
        OUT_DIR / "04_water_sensitivity.png",
        "Water Mask Sensitivity — SAR signal corrupted + water_mask channel set",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 5: Forest cover
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 5: Forest cover")
    fcf_configs = [
        ("FCF = 0%\n(alpine)", 0),
        ("FCF = 15%\n(original-like)", None),
        ("FCF = 50%\n(mixed forest)", 50),
        ("FCF = 85%\n(dense forest)", 85),
    ]
    fcf_scenarios = []
    for label, fcf_val in fcf_configs:
        s = static_orig.copy()
        if fcf_val is not None:
            s[CH["fcf"]] = normalize_static_channel(
                np.full((PATCH, PATCH), float(fcf_val), dtype=np.float32), "fcf"
            )
        p = run_cnn(model, sar_orig, s)
        fcf_display = s[CH["fcf"]] * 100
        fcf_scenarios.append(
            (label, fcf_display, p, mask,
             f"fcf={fcf_display.mean():.0f}%\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        fcf_scenarios,
        OUT_DIR / "05_forest_cover_sensitivity.png",
        "Forest Cover Sensitivity — same SAR signal, overriding fcf channel",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 6: SAR signal strength
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 6: Signal strength")
    # Scale the SAR change channel by different factors
    scale_configs = [
        ("0.25x signal\n(very weak)", 0.25),
        ("0.5x signal\n(weak)", 0.5),
        ("1.0x signal\n(original)", 1.0),
        ("1.5x signal\n(strong)", 1.5),
        ("2.0x signal\n(very strong)", 2.0),
    ]
    sig_scenarios = []
    for label, scale in scale_configs:
        sar_mod = sar_orig.copy()
        # Scale only the change channel (index 0), leave ANF (index 1)
        sar_mod[:, 0, :, :] *= scale
        p = run_cnn(model, sar_mod, static_orig)
        sig_scenarios.append(
            (label, sar_mod[best_track, 0], p, mask,
             f"scale={scale}x\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        sig_scenarios,
        OUT_DIR / "06_signal_strength_sensitivity.png",
        "SAR Signal Strength — scaling backscatter change channel",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 7: Track count
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 7: Track count")
    # Find which tracks are active (have signal)
    active_tracks = []
    for i in range(sar_orig.shape[0]):
        if sar_orig[i, 0].std() > 0.01:
            active_tracks.append(i)
    log.info("Active tracks: %s", active_tracks)

    track_configs = [
        ("Best track only", [best_track]),
        ("2 best tracks", active_tracks[:2] if len(active_tracks) >= 2 else active_tracks),
        (f"All {len(active_tracks)} active", active_tracks),
        (f"All {sar_orig.shape[0]} tracks\n(incl. inactive)", list(range(sar_orig.shape[0]))),
    ]
    track_scenarios = []
    for label, track_ids in track_configs:
        sar_sub = sar_orig[track_ids]
        p = run_cnn(model, sar_sub, static_orig)
        track_scenarios.append(
            (label, sar_sub[0, 0], p, mask,
             f"{len(track_ids)} tracks\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        track_scenarios,
        OUT_DIR / "07_track_count_sensitivity.png",
        "Track Count — how many SAR passes the CNN receives",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 8: ANF (viewing geometry) override
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 8: ANF quality")
    from sarvalanche.ml.v2.patch_extraction import normalize_anf

    anf_configs = [
        ("Original ANF", None),
        ("Excellent\n(raw ANF=1)", 1.0),
        ("Poor\n(raw ANF=30)", 30.0),
        ("Very poor\n(raw ANF=200)", 200.0),
    ]
    anf_scenarios = []
    for label, raw_anf in anf_configs:
        sar_mod = sar_orig.copy()
        if raw_anf is not None:
            anf_norm = normalize_anf(np.full((PATCH, PATCH), raw_anf, dtype=np.float32))
            for t in range(sar_mod.shape[0]):
                sar_mod[t, 1] = anf_norm
        p = run_cnn(model, sar_mod, static_orig)
        anf_display = sar_mod[best_track, 1]
        anf_scenarios.append(
            (label, anf_display, p, mask,
             f"ANF norm mean={anf_display.mean():.2f}\nprob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        anf_scenarios,
        OUT_DIR / "08_anf_sensitivity.png",
        "Viewing Geometry (ANF) — same SAR change, overriding ANF quality channel",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 9: D3.0 miss recreation — apply all degrading factors
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 9: D3.0 miss — cumulative degradation")

    steps = [
        ("Original", {}),
        ("+ slope=21°", {"slope": np.radians(21)}),
        ("+ cc=8", {"slope": np.radians(21), "cc": 8}),
        ("+ 15% water", {"slope": np.radians(21), "cc": 8, "water": 0.15}),
    ]
    degrade_scenarios = []
    rng = np.random.RandomState(77)
    for label, mods in steps:
        s = static_orig.copy()
        sar_mod = sar_orig.copy()

        if "slope" in mods:
            s[CH["slope"]] = normalize_static_channel(
                np.full((PATCH, PATCH), mods["slope"], dtype=np.float32), "slope"
            )
        if "cc" in mods:
            s[CH["cell_counts"]] = normalize_static_channel(
                np.full((PATCH, PATCH), float(mods["cc"]), dtype=np.float32), "cell_counts"
            )
        if "water" in mods:
            wm = np.zeros((PATCH, PATCH), dtype=np.float32)
            wm[rng.random((PATCH, PATCH)) < mods["water"]] = 1.0
            s[CH["water_mask"]] = wm
            for t in range(sar_mod.shape[0]):
                noise = rng.normal(0, 1.5, sar_mod[t, 0][wm > 0.5].shape).astype(np.float32)
                sar_mod[t, 0][wm > 0.5] = noise

        p = run_cnn(model, sar_mod, s)
        degrade_scenarios.append(
            (label, sar_mod[best_track, 0], p, mask,
             f"prob_debris={p[debris].mean():.3f}\nmax_prob={p.max():.3f}")
        )

    plot_comparison_grid(
        degrade_scenarios,
        OUT_DIR / "09_d3_miss_cumulative.png",
        "D3.0 Miss — cumulative terrain degradation on real patch",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 10: Kill the SAR signal entirely — what does terrain alone give?
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 10: SAR vs terrain")
    # Zero out all SAR
    sar_zero = np.zeros_like(sar_orig)
    # Keep ANF channels
    sar_zero[:, 1, :, :] = sar_orig[:, 1, :, :]
    prob_no_sar = run_cnn(model, sar_zero, static_orig)

    # Zero out all static terrain (keep DEM for normalization)
    static_zero = np.zeros_like(static_orig)
    static_zero[CH["dem"]] = static_orig[CH["dem"]]
    prob_no_terrain = run_cnn(model, sar_orig, static_zero)

    # Zero out d_empirical only
    static_no_demp = static_orig.copy()
    static_no_demp[CH["d_empirical"]] = 0.0
    prob_no_demp = run_cnn(model, sar_orig, static_no_demp)

    terrain_scenarios = [
        ("Full inputs\n(baseline)", sar_display, prob_base, mask,
         f"All channels present\nprob_debris={prob_base[debris].mean():.3f}"),
        ("No d_empirical\n(static zeroed)", sar_display, prob_no_demp, mask,
         f"d_empirical=0\nprob_debris={prob_no_demp[debris].mean():.3f}"),
        ("No SAR change\n(terrain only)", np.zeros_like(sar_display), prob_no_sar, mask,
         f"SAR change=0\nprob_debris={prob_no_sar[debris].mean():.3f}"),
        ("No terrain\n(SAR only)", sar_display, prob_no_terrain, mask,
         f"Static=0 (except DEM)\nprob_debris={prob_no_terrain[debris].mean():.3f}"),
    ]

    plot_comparison_grid(
        terrain_scenarios,
        OUT_DIR / "10_sar_vs_terrain.png",
        "Channel Ablation — which inputs drive the CNN?",
    )

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 11: Bottleneck visualization
    # ══════════════════════════════════════════════════════════════════
    log.info("Scenario 11: Bottleneck resolution")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    channel_labels = ["SAR change", "Slope (°)", "Cell counts", "Release zones", "d_empirical"]
    arrays_full = [
        sar_display,
        static_orig[CH["slope"]] * 0.6 * (180 / np.pi),
        np.sign(static_orig[CH["cell_counts"]]) * (
            np.exp(np.abs(static_orig[CH["cell_counts"]]) * 5.0) - 1
        ),
        static_orig[CH["release_zones"]],
        static_orig[CH["d_empirical"]] * 5.0,
    ]
    cmaps = ["RdBu_r", "YlOrRd", "YlOrRd", "Greens", "RdBu_r"]

    for i, (name, arr, cmap) in enumerate(zip(channel_labels, arrays_full, cmaps)):
        im = axes[0, i].imshow(arr, cmap=cmap, origin="upper")
        axes[0, i].contour(mask, levels=[0.5], colors="lime" if i < 3 else "red",
                           linewidths=0.8, linestyles="--")
        axes[0, i].set_title(f"{name}\n(128x128)", fontsize=9)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        fig.colorbar(im, ax=axes[0, i], shrink=0.6)

        small = zoom(arr, 8.0 / 128.0, order=1)
        im2 = axes[1, i].imshow(small, cmap=cmap, origin="upper",
                                interpolation="nearest")
        axes[1, i].set_title(f"{name}\n(8x8 bottleneck)", fontsize=9)
        axes[1, i].set_xticks(range(8))
        axes[1, i].set_yticks(range(8))
        axes[1, i].grid(True, color="gray", linewidth=0.5, alpha=0.5)
        fig.colorbar(im2, ax=axes[1, i], shrink=0.6)

    fig.suptitle(
        "CNN Bottleneck: 128x128 inputs → 8x8 features\n"
        f"Debris deposit ({mask.mean():.1%} of patch, dashed outline) at full vs bottleneck resolution",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "11_bottleneck_resolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: 11_bottleneck_resolution.png")

    log.info("All figures saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
