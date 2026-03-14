"""Assign confidence weights to v2 training patches.

Writes a 'confidence' field (0.0–1.0) into each labels.json entry.
High confidence = trust this label. Low confidence = down-weight in training.

Scoring logic:
  Positive patches:
    - Base confidence from mean d_empirical in debris pixels (higher = better)
    - Penalize low slope in debris region (avalanches need slope)
    - Penalize spring dates (April+) where wet snow confounds signal
    - Very low slope + spring → near-zero confidence (wet snow on flat ground)
  Negative patches:
    - Base confidence 1.0 (most negatives are trustworthy)
    - Reduce if patch has large bright d_empirical regions (possible missed avy)

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/assign_confidence.py \
        --patches-dir local/issw/v2_patches
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

from sarvalanche.ml.v2.channels import STATIC_CHANNELS

D_EMPIRICAL_IDX = STATIC_CHANNELS.index('d_empirical')
D_CR_IDX = STATIC_CHANNELS.index('d_cr') if 'd_cr' in STATIC_CHANNELS else None
SLOPE_IDX = STATIC_CHANNELS.index('slope')

# Dates in spring melt window get penalized
SPRING_MONTHS = {3, 4, 5}  # March gets mild penalty, April+ gets heavy


def _parse_date_from_path(date_dir: Path) -> str | None:
    """Extract YYYY-MM-DD date string from path."""
    for part in reversed(date_dir.parts):
        if re.match(r"\d{4}-\d{2}-\d{2}", part):
            return part
    return None


def score_positive(static, label_mask, date_str: str | None) -> float:
    """Confidence score for a positive patch."""
    d_emp = static[D_EMPIRICAL_IDX]
    slope = static[SLOPE_IDX]
    debris_px = label_mask > 0.5
    n_debris = debris_px.sum()

    if n_debris < 3:
        return 0.1  # almost no debris pixels → very suspicious

    mean_d = float(np.nanmean(d_emp[debris_px]))
    mean_slope = float(np.nanmean(slope[debris_px]))

    # Cross-ratio change in debris region (if available)
    mean_d_cr = None
    if D_CR_IDX is not None and D_CR_IDX < static.shape[0]:
        d_cr = static[D_CR_IDX]
        mean_d_cr = float(np.nanmean(d_cr[debris_px]))

    # --- d_empirical signal confidence (0.2 to 1.0) ---
    # Higher mean d_empirical in debris region = more confident
    # Typical good values: 0.2-0.5 (after log1p normalization)
    d_conf = np.clip((mean_d - 0.05) / 0.4, 0.2, 1.0)

    # --- Slope confidence (0.1 to 1.0) ---
    # Slope is normalized by /0.6, so 0.15 ≈ 9°, 0.25 ≈ 15°, 0.5 ≈ 30°
    # Avalanche debris should be on some slope (even runout zones are typically >5°)
    if mean_slope < 0.08:  # < ~5° very flat
        slope_conf = 0.1
    elif mean_slope < 0.15:  # < ~9° quite flat
        slope_conf = 0.4
    elif mean_slope < 0.25:  # < ~15° gentle
        slope_conf = 0.7
    else:
        slope_conf = 1.0

    # --- Seasonal penalty ---
    season_conf = 1.0
    if date_str:
        month = int(date_str.split("-")[1])
        if month == 4:
            season_conf = 0.4  # April: heavy melt penalty
        elif month >= 5:
            season_conf = 0.2  # May+: extreme melt
        elif month == 3 and int(date_str.split("-")[2]) > 10:
            season_conf = 0.7  # Late March: mild penalty
        elif month == 11 and int(date_str.split("-")[2]) < 20:
            season_conf = 0.8  # Early November: mild early-season penalty

    # --- Combine: geometric mean, with floor ---
    # Flat ground + spring = extremely low confidence (wet snow)
    conf = (d_conf * slope_conf * season_conf) ** (1 / 3)

    # --- Cross-ratio penalty (wet snow indicator) ---
    # Negative d_cr in debris region suggests wet snow (VH drops more than VV)
    # Avalanche debris tends to have neutral or positive d_cr
    cr_conf = 1.0
    if mean_d_cr is not None:
        if mean_d_cr < -0.3:
            cr_conf = 0.3  # strongly negative = likely wet snow
        elif mean_d_cr < -0.1:
            cr_conf = 0.6  # mildly negative = suspicious

    # --- Combine: geometric mean, with floor ---
    # Flat ground + spring = extremely low confidence (wet snow)
    conf = (d_conf * slope_conf * season_conf * cr_conf) ** (1 / 4)

    # Extra penalty: flat + spring = almost certainly wet snow
    if mean_slope < 0.15 and date_str:
        month = int(date_str.split("-")[1])
        if month >= 4:
            conf = min(conf, 0.05)
        elif month == 3 and int(date_str.split("-")[2]) > 10:
            conf = min(conf, 0.2)

    # Extra penalty: negative CR + spring = wet snow
    if mean_d_cr is not None and mean_d_cr < -0.2 and date_str:
        month = int(date_str.split("-")[1])
        if month >= 3:
            conf = min(conf, 0.15)

    return float(np.clip(conf, 0.05, 1.0))


def score_negative(static, date_str: str | None) -> float:
    """Confidence score for a negative patch."""
    d_emp = static[D_EMPIRICAL_IDX]

    # Fraction of patch with high d_empirical
    high_frac = float((d_emp > 0.5).sum()) / d_emp.size
    max_d = float(np.nanmax(d_emp))

    # Most negatives are fine
    conf = 1.0

    # Penalize if large bright regions (possible missed detection)
    if high_frac > 0.15:
        conf = 0.3
    elif high_frac > 0.05:
        conf = 0.5
    elif max_d > 0.7:
        conf = 0.7

    # Spring dates: bright negatives are MORE trustworthy (correctly excluded melt)
    # so actually boost confidence back up for spring
    if date_str:
        month = int(date_str.split("-")[1])
        if month >= 4 and high_frac > 0.05:
            # Bright negative in spring = probably correct (it's melt, not debris)
            conf = max(conf, 0.8)

    return float(conf)


def process_date_dir(date_dir: Path) -> tuple[int, int]:
    """Score all patches in a date directory, update labels.json."""
    labels_path = date_dir / "labels.json"
    if not labels_path.exists():
        return 0, 0

    with open(labels_path) as f:
        labels = json.load(f)

    if not labels:
        return 0, 0

    date_str = _parse_date_from_path(date_dir)
    updated = 0

    for patch_id, info in labels.items():
        npz_path = date_dir / f"{patch_id}_v2_.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        static = data["static"]
        label_mask = data["label_mask"] if "label_mask" in data else None

        if info["label"] == 1:
            if label_mask is not None:
                conf = score_positive(static, label_mask, date_str)
            else:
                conf = 0.5  # no mask → moderate confidence
        else:
            conf = score_negative(static, date_str)

        info["confidence"] = round(conf, 4)
        updated += 1

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    confs = [v["confidence"] for v in labels.values() if "confidence" in v]
    return updated, len([c for c in confs if c < 0.5])


def main():
    parser = argparse.ArgumentParser(description="Assign confidence weights to patches")
    parser.add_argument("--patches-dir", type=Path, default=Path("local/issw/v2_patches"))
    args = parser.parse_args()

    label_files = sorted(args.patches_dir.rglob("labels.json"))
    label_files = [f for f in label_files if "_corrupted" not in str(f)]

    log.info("Found %d label files", len(label_files))

    total_updated = 0
    total_low = 0
    for lf in label_files:
        n_updated, n_low = process_date_dir(lf.parent)
        rel = str(lf.parent.relative_to(args.patches_dir))
        if n_updated > 0:
            log.info("  %s: %d scored, %d low-confidence (<0.5)", rel, n_updated, n_low)
        total_updated += n_updated
        total_low += n_low

    log.info("Done. %d patches scored, %d low-confidence", total_updated, total_low)


if __name__ == "__main__":
    main()
