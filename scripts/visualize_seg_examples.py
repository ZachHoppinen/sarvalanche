#!/usr/bin/env python
"""
Visualize CNN segmentation predictions on example tracks labeled as 3 (high confidence debris).

For each example, shows a 2x3 grid:
  Row 1: input channels — Mahalanobis distance, empirical p-value, track mask
  Row 2: segmentation target (p_pixelwise + shapes), CNN prediction (sigmoid), overlay

Usage
-----
    conda run -n sarvalanche python scripts/visualize_seg_examples.py
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sarvalanche.ml.track_classifier import BINARY_THRESHOLD, _load_ds, build_seg_training_set
from sarvalanche.ml.track_features import extract_track_patch_with_target
from sarvalanche.ml.track_patch_encoder import CNN_SEG_ENCODER_PATH, TrackSegEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

RUNS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')
LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
SHAPES_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')
OUT_DIR     = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/seg_visualizations')

MAX_EXAMPLES = 8  # show at most this many tracks


def main() -> None:
    # ── Load model ────────────────────────────────────────────────────────
    if not CNN_SEG_ENCODER_PATH.exists():
        log.error("No saved model at %s — run train_track_encoder.py first.", CNN_SEG_ENCODER_PATH)
        return

    model = TrackSegEncoder()
    model.load_state_dict(torch.load(CNN_SEG_ENCODER_PATH, map_location='cpu', weights_only=True))
    model.eval()
    log.info("Loaded model from %s", CNN_SEG_ENCODER_PATH)

    # ── Load labels, keep only label == 3 ─────────────────────────────────
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    label3 = {k: v for k, v in all_labels.items() if v['label'] == 3}
    if not label3:
        log.error("No tracks with label == 3 found in %s", LABELS_PATH)
        return
    log.info("Found %d tracks with label == 3", len(label3))

    # ── Build patches with targets (includes shapes blending) ─────────────
    patches, targets, y = build_seg_training_set(
        label3, RUNS_DIR, shapes_path=SHAPES_PATH,
    )
    keys = list(label3.keys())
    log.info("Built %d patches", len(patches))

    # ── Run inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        seg_logits = model.segment(torch.FloatTensor(patches))  # (N, 1, H, W)
        seg_probs = torch.sigmoid(seg_logits).numpy()[:, 0]     # (N, H, W)

    # ── Plot ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = min(len(patches), MAX_EXAMPLES)

    for i in range(n):
        key = keys[i]
        patch = patches[i]       # (C, H, W)
        target = targets[i, 0]   # (H, W)
        pred = seg_probs[i]      # (H, W)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f'{key}', fontsize=13)

        # Row 1: input channels
        ax = axes[0, 0]
        im = ax.imshow(patch[0], cmap='plasma', vmin=0.2, vmax=0.7)
        ax.set_title('Mahalanobis Distance')
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 1]
        im = ax.imshow(patch[1], cmap='RdYlGn_r', vmin=0.2, vmax=0.7)
        ax.set_title('Empirical p-value')
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 2]
        im = ax.imshow(patch[3], cmap='bone',
                        vmin=np.deg2rad(15), vmax=np.deg2rad(45))
        ax.contour(patch[6], levels=[0.5], colors='red', linewidths=1.0)
        ax.set_title('Slope (rad) + Track Outline')
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: target, prediction, overlay
        ax = axes[1, 0]
        im = ax.imshow(target, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Target (p_pixelwise + shapes)')
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 1]
        im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        ax.set_title('CNN Prediction (sigmoid)')
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Overlay: prediction contours on Mahalanobis background
        ax = axes[1, 2]
        ax.imshow(patch[0], cmap='plasma', vmin=0.2, vmax=0.7)
        ax.contour(pred, levels=[0.3, 0.5, 0.7], colors=['cyan', 'yellow', 'red'],
                   linewidths=1.2)
        ax.contour(patch[6], levels=[0.5], colors='white', linewidths=0.8,
                   linestyles='dashed')
        ax.set_title('Prediction contours on Mahalanobis')

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        safe_key = key.replace('|', '_')
        out_path = OUT_DIR / f'seg_{safe_key}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info("  [%d/%d] saved %s", i + 1, n, out_path.name)

    log.info("Done — %d figures saved to %s", n, OUT_DIR)


if __name__ == '__main__':
    main()
