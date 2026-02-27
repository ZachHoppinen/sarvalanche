"""
export_weights.py

Exports a PyTorch checkpoint (.pth) to sarvalanche/ml/weights/ with
version-stamped filenames and a matching .txt sidecar containing training
metadata extracted automatically from the checkpoint dict.

Usage
-----
    python export_weights.py --help

Example
-------
    python export_weights.py \
        --checkpoint /tmp/model_checkpoint.pth \
        --name avalanche_detector \
        --train-samples 1200 \
        --test-samples 300 \
        --metrics '{"precision": 0.91, "recall": 0.85, "f1": 0.88}' \
        --notes "Trained on 2023-2024 Cascades dataset"
"""

import argparse
import importlib.metadata
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

log = logging.getLogger(__name__)


WEIGHTS_DIR = Path(__file__).parent / "weights" / "rtc_predictor"


def get_sarvalanche_version() -> str:
    try:
        return importlib.metadata.version("sarvalanche")
    except importlib.metadata.PackageNotFoundError:
        log.warning("sarvalanche package not found, using 'unknown' for version")
        return "unknown"


def build_filename(model_name: str, version: str) -> str:
    """Build a versioned filename stem e.g. avalanche_detector_v0.3.1_20240315"""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{model_name}_v{version}_{date_str}"


def extract_checkpoint_metadata(checkpoint: dict) -> dict:
    """Pull known useful fields out of a sarvalanche checkpoint dict."""
    meta = {}

    # Training state
    for key in ("epoch", "train_loss", "val_loss"):
        if key in checkpoint:
            meta[key] = checkpoint[key]

    # Model architecture
    if "model_config" in checkpoint:
        meta["model_config"] = checkpoint["model_config"]

    # Data config
    if "zones" in checkpoint:
        meta["zones"] = checkpoint["zones"]
    if "seasons" in checkpoint:
        meta["seasons"] = checkpoint["seasons"]

    return meta


def write_sidecar(
    path: Path,
    model_name: str,
    version: str,
    train_samples: int,
    test_samples: int,
    checkpoint_meta: dict,
    extra_metrics: dict | None,
    notes: str | None,
) -> None:
    lines = [
        "=" * 60,
        "sarvalanche model export",
        "=" * 60,
        f"model name       : {model_name}",
        f"sarvalanche ver  : {version}",
        f"exported at (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "--- training state ---",
        f"epoch            : {checkpoint_meta.get('epoch', 'n/a')}",
        f"train loss       : {checkpoint_meta.get('train_loss', 'n/a')}",
        f"val loss         : {checkpoint_meta.get('val_loss', 'n/a')}",
        "",
        "--- training data ---",
        f"train samples    : {train_samples}",
        f"test samples     : {test_samples}",
        f"total samples    : {train_samples + test_samples}",
    ]

    if "zones" in checkpoint_meta:
        lines.append(f"zones            : {checkpoint_meta['zones']}")
    if "seasons" in checkpoint_meta:
        lines.append(f"seasons          : {checkpoint_meta['seasons']}")

    if "model_config" in checkpoint_meta:
        lines += ["", "--- model config ---"]
        for k, v in checkpoint_meta["model_config"].items():
            lines.append(f"  {k:<16}: {v}")

    if extra_metrics:
        lines += ["", "--- evaluation metrics ---"]
        for k, v in extra_metrics.items():
            lines.append(f"  {k:<16}: {v}")

    if notes:
        lines += ["", "--- notes ---", notes]

    lines += ["", "=" * 60]

    path.write_text("\n".join(lines) + "\n")
    log.info("Sidecar written: %s", path)


def export_weights(
    checkpoint_path: Path,
    model_name: str,
    train_samples: int,
    test_samples: int,
    extra_metrics: dict | None = None,
    notes: str | None = None,
    weights_dir: Path = WEIGHTS_DIR,
) -> Path:
    """
    Copy a checkpoint .pth and write a metadata sidecar to weights_dir.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the .pth checkpoint file produced during training.
    model_name : str
        Short descriptive name, e.g. 'avalanche_detector'.
    train_samples : int
        Number of training samples used.
    test_samples : int
        Number of test samples used.
    extra_metrics : dict, optional
        Evaluation metrics not already in the checkpoint,
        e.g. {'precision': 0.91, 'recall': 0.85}.
    notes : str, optional
        Free-text notes to include in the sidecar.
    weights_dir : Path
        Directory to write files into. Created if it doesn't exist.

    Returns
    -------
    dest : Path
        Path to the copied .pth file in weights_dir.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    weights_dir.mkdir(parents=True, exist_ok=True)

    version = get_sarvalanche_version()
    stem = build_filename(model_name, version)

    dest = weights_dir / f"{stem}.pth"
    sidecar_path = weights_dir / f"{stem}.txt"

    if dest.exists():
        response = input(f"{dest.name} already exists. Overwrite? [y/N]: ")
        if response.strip().lower() != "y":
            log.info("Export cancelled.")
            sys.exit(0)

    # Load checkpoint to extract metadata (cpu-safe)
    log.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = extract_checkpoint_metadata(checkpoint)

    shutil.copy2(checkpoint_path, dest)
    log.info("Checkpoint copied: %s", dest)

    write_sidecar(
        path=sidecar_path,
        model_name=model_name,
        version=version,
        train_samples=train_samples,
        test_samples=test_samples,
        checkpoint_meta=checkpoint_meta,
        extra_metrics=extra_metrics,
        notes=notes,
    )

    return dest


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Export sarvalanche model weights")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--name", required=True,
                        help="Short model name, e.g. avalanche_detector")
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--test-samples", type=int, required=True)
    parser.add_argument("--metrics", type=json.loads, default=None,
                        help='JSON string e.g. \'{"precision": 0.91, "recall": 0.85}\'')
    parser.add_argument("--notes", default=None,
                        help="Optional free-text notes")
    parser.add_argument("--weights-dir", default=str(WEIGHTS_DIR),
                        help="Override output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    export_weights(
        checkpoint_path=Path(args.checkpoint),
        model_name=args.name,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        extra_metrics=args.metrics,
        notes=args.notes,
        weights_dir=Path(args.weights_dir),
    )