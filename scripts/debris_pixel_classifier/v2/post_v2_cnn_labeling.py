"""Post-labeling: extract patches, train CNN, and run inference for all labeled dates.

Auto-discovers dates that have both:
  - avalanche_labels_<date>.gpkg  (debris polygons from QGIS)
  - geotiffs/<date>/              (footprint windows)

Then for each date:
  1. Extract v2 training patches (128x128) from polygons within footprint windows
  2. Train DebrisDetector on all extracted patches
  3. Run sliding-window inference for each date → scene probability GeoTIFF

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/post_v2_cnn_labeling.py \
        --nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
        --labels-dir local/issw/debris_shapes \
        --patches-dir local/issw/v2_patches \
        --epochs 50 --pos-weight 10 --batch-size 4

    # Skip training (use existing weights) and just run inference:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/post_v2_cnn_labeling.py \
        --nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
        --labels-dir local/issw/debris_shapes \
        --patches-dir local/issw/v2_patches \
        --skip-extract --skip-train \
        --weights local/issw/v2_patches/v2_detector_best.pt
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

TAU = 6


def discover_labeled_dates(labels_dir: Path, season_start: str, season_end: str) -> list[str]:
    """Find dates that have both a labels gpkg and a geotiff directory.

    Only includes dates within the season window (season_start to season_end).
    """
    dates = []
    for gpkg in sorted(labels_dir.glob("avalanche_labels_*.gpkg")):
        # Extract date from filename: avalanche_labels_YYYY-MM-DD.gpkg
        stem = gpkg.stem  # avalanche_labels_YYYY-MM-DD
        date_str = stem.replace("avalanche_labels_", "")

        # Validate it's a real date
        try:
            dt = pd.Timestamp(date_str)
        except ValueError:
            continue

        # Check within season bounds
        if dt < pd.Timestamp(season_start) or dt > pd.Timestamp(season_end):
            log.info("  Skipping %s (outside season %s to %s)", date_str, season_start, season_end)
            continue

        # Check geotiff directory exists
        geotiff_dir = labels_dir / "geotiffs" / date_str
        if not geotiff_dir.is_dir() or not list(geotiff_dir.glob("*.tif")):
            log.warning("  Skipping %s (no GeoTIFFs in %s)", date_str, geotiff_dir)
            continue

        dates.append(date_str)

    return dates


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a subprocess command, return True on success."""
    log.info("Running: %s", description)
    log.info("  %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("  FAILED (exit %d): %s", result.returncode, description)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Post-labeling: extract patches, train CNN, run inference",
    )
    parser.add_argument(
        "--nc", type=Path, required=True,
        help="Path to season_dataset.nc",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("local/issw/debris_shapes"),
        help="Directory with avalanche_labels_<date>.gpkg and geotiffs/ (default: local/issw/debris_shapes)",
    )
    parser.add_argument(
        "--patches-dir", type=Path, default=Path("local/issw/v2_patches"),
        help="Directory for extracted patches (default: local/issw/v2_patches)",
    )
    parser.add_argument(
        "--season", type=str, default="2024-2025",
        help='Season "YYYY-YYYY" to filter dates (default: 2024-2025)',
    )
    parser.add_argument(
        "--tau", type=float, default=TAU,
        help=f"Temporal decay tau in days (default: {TAU})",
    )

    # Patch extraction args
    parser.add_argument("--stride", type=int, default=64, help="Extraction stride (default: 64)")
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="Neg:pos patch ratio (default: 3.0)")
    parser.add_argument("--min-debris-frac", type=float, default=0.005, help="Min debris fraction (default: 0.005)")

    # Training args
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--pos-weight", type=float, default=0.0, help="Pos weight (0=auto, default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device override")

    # Inference args
    parser.add_argument("--inference-stride", type=int, default=32, help="Inference stride (default: 32)")
    parser.add_argument("--inference-batch-size", type=int, default=16, help="Inference batch size (default: 16)")

    # Skip flags
    parser.add_argument("--skip-extract", action="store_true", help="Skip patch extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path to model weights (default: <patches-dir>/v2_detector_best.pt)",
    )
    args = parser.parse_args()

    # Parse season
    try:
        start_year, end_year = args.season.split("-")
        season_start = f"{start_year}-11-01"
        season_end = f"{end_year}-04-30"
    except ValueError:
        parser.error('--season must be "YYYY-YYYY"')

    # ── Discover labeled dates ───────────────────────────────────────────
    log.info("Discovering labeled dates in %s (season %s to %s)", args.labels_dir, season_start, season_end)
    dates = discover_labeled_dates(args.labels_dir, season_start, season_end)
    if not dates:
        log.error("No labeled dates found. Run pre_v2_cnn_labeling.py first, then label in QGIS.")
        return

    log.info("Found %d labeled dates: %s", len(dates), ", ".join(dates))

    # ── 1. Extract patches for each date ─────────────────────────────────
    if not args.skip_extract:
        log.info("")
        log.info("=" * 60)
        log.info("STEP 1: Extract training patches")
        log.info("=" * 60)

        for date_str in dates:
            polygons_path = args.labels_dir / f"avalanche_labels_{date_str}.gpkg"
            geotiff_dir = args.labels_dir / "geotiffs" / date_str
            out_dir = args.patches_dir / date_str

            log.info("")
            log.info("--- Extracting patches for %s ---", date_str)

            cmd = [
                sys.executable,
                "scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py",
                "--nc", str(args.nc),
                "--polygons", str(polygons_path),
                "--geotiff-dir", str(geotiff_dir),
                "--date", date_str,
                "--tau", str(args.tau),
                "--out-dir", str(out_dir),
                "--stride", str(args.stride),
                "--neg-ratio", str(args.neg_ratio),
                "--min-debris-frac", str(args.min_debris_frac),
            ]

            if not run_cmd(cmd, f"extract patches for {date_str}"):
                log.warning("Extraction failed for %s, continuing...", date_str)
    else:
        log.info("Skipping patch extraction (--skip-extract)")

    # ── 2. Train ─────────────────────────────────────────────────────────
    weights_path = args.weights or (args.patches_dir / "v2_detector_best.pt")

    if not args.skip_train:
        log.info("")
        log.info("=" * 60)
        log.info("STEP 2: Train DebrisDetector")
        log.info("=" * 60)

        cmd = [
            sys.executable,
            "scripts/debris_pixel_classifier/v2/train.py",
            "--data-dir", str(args.patches_dir),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
            "--pos-weight", str(args.pos_weight),
            "--out", str(weights_path),
        ]
        if args.device:
            cmd += ["--device", args.device]

        if not run_cmd(cmd, "train DebrisDetector"):
            log.error("Training failed. Fix errors and re-run, or use --skip-train with --weights.")
            return
    else:
        log.info("Skipping training (--skip-train)")

    # ── 3. Inference for each date ───────────────────────────────────────
    if not args.skip_inference:
        log.info("")
        log.info("=" * 60)
        log.info("STEP 3: Run inference for each date")
        log.info("=" * 60)

        if not weights_path.exists():
            log.error("Weights not found: %s. Train first or provide --weights.", weights_path)
            return

        inference_out_dir = args.nc.parent / "v2_inference"
        inference_out_dir.mkdir(parents=True, exist_ok=True)

        for date_str in dates:
            out_tif = inference_out_dir / f"scene_v2_debris_{date_str}.tif"

            log.info("")
            log.info("--- Inference for %s ---", date_str)

            cmd = [
                sys.executable,
                "scripts/debris_pixel_classifier/v2/inference_scene.py",
                "--nc", str(args.nc),
                "--date", date_str,
                "--tau", str(args.tau),
                "--weights", str(weights_path),
                "--stride", str(args.inference_stride),
                "--batch-size", str(args.inference_batch_size),
                "--out", str(out_tif),
            ]
            if args.device:
                cmd += ["--device", args.device]

            if not run_cmd(cmd, f"inference for {date_str}"):
                log.warning("Inference failed for %s, continuing...", date_str)
    else:
        log.info("Skipping inference (--skip-inference)")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    log.info("Dates processed: %s", ", ".join(dates))
    if not args.skip_extract:
        log.info("Patches: %s", args.patches_dir)
    if not args.skip_train:
        log.info("Model weights: %s", weights_path)
    if not args.skip_inference:
        inference_out_dir = args.nc.parent / "v2_inference"
        log.info("Inference outputs: %s", inference_out_dir)
        for date_str in dates:
            out_tif = inference_out_dir / f"scene_v2_debris_{date_str}.tif"
            if out_tif.exists():
                log.info("  %s", out_tif)


if __name__ == "__main__":
    main()
