"""Review v2 training patches: render montages for visual QA.

Generates PNG montages of patches sorted by suspiciousness:
  - Positive patches with low mean d_empirical in debris pixels
  - Negative patches with high mean d_empirical (possible misses)

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/review_patches.py \
        --patches-dir local/issw/v2_patches \
        --out-dir local/v2_review
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

from sarvalanche.ml.v2.channels import STATIC_CHANNELS

D_EMPIRICAL_IDX = STATIC_CHANNELS.index('d_empirical')
SLOPE_IDX = STATIC_CHANNELS.index('slope')
DEM_IDX = STATIC_CHANNELS.index('dem')
PATCH_SIZE = 128

# Colormap for d_empirical
CMAP_D = LinearSegmentedColormap.from_list("d_emp", ["#1a1a2e", "#16213e", "#e94560", "#ffdd00"])


def load_patch(npz_path):
    """Load a patch npz and return static channels, label_mask, label."""
    data = np.load(npz_path)
    return {
        "static": data["static"],
        "label_mask": data["label_mask"],
        "label": int(data["label"]),
    }


def compute_suspiciousness(static, label_mask, label):
    """Score how suspicious a label is. Higher = more suspicious."""
    d_emp = static[D_EMPIRICAL_IDX]
    slope = static[SLOPE_IDX]

    if label == 1:
        # Positive patch: suspicious if debris pixels have LOW d_empirical
        debris_px = label_mask > 0.5
        if debris_px.sum() < 3:
            return 1.0  # very few debris pixels is suspicious
        mean_d_in_debris = float(np.nanmean(d_emp[debris_px]))
        mean_slope_in_debris = float(np.nanmean(slope[debris_px]))
        # Low d_empirical in debris = suspicious (should be high)
        # Very low slope in debris = suspicious (avalanches need slope)
        score = 0.0
        if mean_d_in_debris < 0.3:
            score += 0.5
        if mean_d_in_debris < 0.1:
            score += 0.3
        if mean_slope_in_debris < 0.15:  # ~9 degrees (normalized by 0.6)
            score += 0.2
        return score
    else:
        # Negative patch: suspicious if there's HIGH d_empirical anywhere
        # (possible missed detection)
        max_d = float(np.nanmax(d_emp))
        high_frac = float((d_emp > 0.5).sum()) / (PATCH_SIZE * PATCH_SIZE)
        score = 0.0
        if max_d > 0.7:
            score += 0.3
        if high_frac > 0.05:
            score += 0.4
        if high_frac > 0.15:
            score += 0.3
        return score


def render_patch(ax, static, label_mask, label, patch_id, suspiciousness):
    """Render a single patch on an axes."""
    d_emp = static[D_EMPIRICAL_IDX]
    slope = static[SLOPE_IDX]

    # Composite: d_empirical as color, slope as brightness modulation
    rgb = CMAP_D(np.clip(d_emp, 0, 1))[:, :, :3]
    brightness = np.clip(slope * 1.5, 0.3, 1.0)
    rgb = rgb * brightness[:, :, None]

    ax.imshow(rgb, interpolation="nearest")

    # Overlay label mask contour
    if label == 1 and label_mask.max() > 0:
        ax.contour(label_mask, levels=[0.5], colors=["lime"], linewidths=1.0)

    # Border color based on suspiciousness
    border_color = "red" if suspiciousness > 0.5 else ("orange" if suspiciousness > 0.2 else "white")
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)

    debris_frac = float((label_mask > 0.5).sum()) / (PATCH_SIZE * PATCH_SIZE) if label == 1 else 0
    ax.set_title(
        f"{patch_id}\ndf={debris_frac:.3f} s={suspiciousness:.2f}",
        fontsize=5,
        color=border_color,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def process_date_dir(date_dir, out_dir, n_cols=6):
    """Process all patches in a date directory, render montages."""
    labels_path = date_dir / "labels.json"
    if not labels_path.exists():
        return

    with open(labels_path) as f:
        labels = json.load(f)

    if not labels:
        return

    # Load all patches
    patches = []
    for patch_id, info in labels.items():
        npz_path = date_dir / f"{patch_id}_v2_.npz"
        if not npz_path.exists():
            continue
        data = load_patch(npz_path)
        susp = compute_suspiciousness(data["static"], data["label_mask"], data["label"])
        patches.append({
            "patch_id": patch_id,
            "label": data["label"],
            "static": data["static"],
            "label_mask": data["label_mask"],
            "suspiciousness": susp,
            "debris_frac": info.get("debris_frac", 0),
        })

    if not patches:
        return

    # Split into positive and negative
    pos_patches = sorted(
        [p for p in patches if p["label"] == 1],
        key=lambda p: -p["suspiciousness"],
    )
    neg_suspicious = sorted(
        [p for p in patches if p["label"] == 0 and p["suspiciousness"] > 0.2],
        key=lambda p: -p["suspiciousness"],
    )

    date_name = date_dir.name
    parent_name = date_dir.parent.name if date_dir.parent.name != "v2_patches" else ""
    prefix = f"{parent_name}_{date_name}" if parent_name else date_name

    log.info(
        "  %s: %d pos patches, %d suspicious negatives",
        prefix, len(pos_patches), len(neg_suspicious),
    )

    # Render positive patches montage
    if pos_patches:
        _render_montage(
            pos_patches, out_dir / f"{prefix}_pos_review.png",
            f"POSITIVE patches — {prefix} ({len(pos_patches)} patches)",
            n_cols,
        )

    # Render suspicious negatives
    if neg_suspicious:
        _render_montage(
            neg_suspicious[:36], out_dir / f"{prefix}_neg_suspicious.png",
            f"SUSPICIOUS NEGATIVES — {prefix} ({len(neg_suspicious)} patches)",
            n_cols,
        )


def _render_montage(patches, out_path, title, n_cols):
    """Render a grid montage of patches."""
    n = len(patches)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.3))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, patch in enumerate(patches):
        render_patch(
            axes_flat[i],
            patch["static"],
            patch["label_mask"],
            patch["label"],
            patch["patch_id"],
            patch["suspiciousness"],
        )

    # Hide unused axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Review v2 training patches")
    parser.add_argument("--patches-dir", type=Path, default=Path("local/issw/v2_patches"))
    parser.add_argument("--out-dir", type=Path, default=Path("local/v2_review"))
    parser.add_argument("--n-cols", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Find all labels.json files
    label_files = sorted(args.patches_dir.rglob("labels.json"))
    label_files = [f for f in label_files if "_corrupted" not in str(f)]

    log.info("Found %d date directories with labels", len(label_files))

    for lf in label_files:
        process_date_dir(lf.parent, args.out_dir, args.n_cols)

    log.info("Done. Review images in %s", args.out_dir)


if __name__ == "__main__":
    main()
