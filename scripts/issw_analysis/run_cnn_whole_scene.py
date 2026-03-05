"""
run_cnn_whole_scene.py — Run the DebrisSegmenter CNN over an entire scene.

Slides a 64×64 window across the scene at 1/4 patch stride (16 pixels),
classifies each patch, and averages overlapping predictions into a
full-scene probability map. Exports as GeoTIFF.

Required dataset variables: d_empirical, fcf, slope, cell_counts,
    release_zones, runout_angle, water_mask.
Optional: combined_distance (zeros if absent).

Usage:
    conda run -n sarvalanche python scripts/issw_analysis/run_cnn_whole_scene.py \
        --nc local/issw/dual_tau_output/zone/season_dataset.nc \
        --date 2025-02-04 \
        --tau 6 \
        --out scene_debris_prob.tif
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr
import pandas as pd

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.debris_segmenter import DebrisSegmenter
from sarvalanche.ml.track_patch_extraction import (
    N_PATCH_CHANNELS,
    _PATCH_DATA_VARS,
    _CHANNEL_NORM,
    _NORTHING_CH,
    _EASTING_CH,
    TRACK_MASK_CHANNEL,
)
from sarvalanche.ml.weight_utils import find_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_channel(arr: np.ndarray, var: str) -> np.ndarray:
    cfg = _CHANNEL_NORM.get(var)
    if not cfg:
        return arr
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return arr


def compute_empirical_for_date(ds, reference_date, tau_days):
    """Compute d_empirical and p_empirical for a given date."""
    from sarvalanche.weights.temporal import get_temporal_weights
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )

    ref_ts = np.datetime64(reference_date)

    # Clean stale variables
    stale_patterns = [
        re.compile(r"^p_\d+_V[VH]_empirical$"),
        re.compile(r"^d_\d+_V[VH]_empirical$"),
    ]
    stale_exact = {"p_empirical", "d_empirical", "w_temporal"}
    to_drop = [v for v in ds.data_vars if v in stale_exact]
    for pat in stale_patterns:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        ds = ds.drop_vars(to_drop)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_empirical, d_empirical = calculate_empirical_backscatter_probability(
        ds,
        ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )

    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical
    return ds


def build_input_stack(ds: xr.Dataset) -> np.ndarray:
    """Build (C, H, W) float32 array from dataset variables.

    Channels match PATCH_CHANNELS order. Position channels (northing, easting)
    are set to 0 (no track context). Track mask is set to 1 (all pixels active).
    """
    H = ds.sizes['y']
    W = ds.sizes['x']
    stack = np.zeros((N_PATCH_CHANNELS, H, W), dtype=np.float32)

    for ch, var in enumerate(_PATCH_DATA_VARS):
        if var in ds.data_vars:
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            stack[ch] = _normalize_channel(arr, var)

    # Position channels: global [-1, +1] grid
    stack[_NORTHING_CH] = np.linspace(1, -1, H, dtype=np.float32)[:, np.newaxis] * np.ones(W, dtype=np.float32)
    stack[_EASTING_CH] = np.ones(H, dtype=np.float32)[:, np.newaxis] * np.linspace(-1, 1, W, dtype=np.float32)

    # Track mask: all 1 (whole scene)
    stack[TRACK_MASK_CHANNEL] = 1.0

    return stack


def sliding_window_inference(
    stack: np.ndarray,
    model: torch.nn.Module,
    patch_size: int = 64,
    stride: int = 16,
    batch_size: int = 64,
    device: str = 'cpu',
) -> np.ndarray:
    """Run CNN over (C, H, W) stack with sliding window.

    Returns (H, W) probability map averaged over overlapping patches.
    """
    C, H, W = stack.shape
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    # Collect patch coordinates
    coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((y0, x0))

    # Handle right and bottom edges
    if (H - patch_size) % stride != 0:
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((H - patch_size, x0))
    if (W - patch_size) % stride != 0:
        for y0 in range(0, H - patch_size + 1, stride):
            coords.append((y0, W - patch_size))
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        coords.append((H - patch_size, W - patch_size))

    # Deduplicate
    coords = list(dict.fromkeys(coords))

    log.info(
        "Sliding window: %d patches (%d×%d scene, patch=%d, stride=%d)",
        len(coords), H, W, patch_size, stride,
    )

    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = np.stack([
                stack[:, y0:y0 + patch_size, x0:x0 + patch_size]
                for y0, x0 in batch_coords
            ])
            x = torch.from_numpy(patches).float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for j, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[j]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

            if (i // batch_size) % 50 == 0:
                log.info("  %d / %d patches processed", min(i + batch_size, len(coords)), len(coords))

    # Average overlapping predictions
    mask = count > 0
    result = np.zeros((H, W), dtype=np.float32)
    result[mask] = (prob_sum[mask] / count[mask]).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run DebrisSegmenter CNN over full scene"
    )
    parser.add_argument("--nc", type=Path, required=True, help="Path to season_dataset.nc")
    parser.add_argument("--date", type=str, required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--tau", type=float, default=6, help="Temporal decay tau in days (default: 6)")
    parser.add_argument("--out", type=Path, default=None, help="Output GeoTIFF path (default: <nc_dir>/scene_debris_<date>.tif)")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride in pixels (default: 16 = patch_size/4)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log.info("Using device: %s", device)

    # Load dataset
    log.info("Loading dataset: %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        log.info("Loading dataset into memory...")
        ds = ds.load()
    log.info("  %d time steps, %d×%d spatial", len(ds.time), ds.sizes['y'], ds.sizes['x'])

    # Ensure w_resolution exists
    if "w_resolution" not in ds.data_vars:
        from sarvalanche.weights.local_resolution import get_local_resolution_weights
        log.info("Computing resolution weights...")
        ds["w_resolution"] = get_local_resolution_weights(ds["anf"])
        ds["w_resolution"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    # Compute empirical layers
    ref_date = pd.Timestamp(args.date)
    log.info("Computing empirical layers for %s (tau=%gd)...", args.date, args.tau)
    ds = compute_empirical_for_date(ds, ref_date, args.tau)

    # Build input stack
    log.info("Building input stack (%d channels)...", N_PATCH_CHANNELS)
    stack = build_input_stack(ds)
    log.info("  Stack shape: %s, range: [%.3f, %.3f]", stack.shape, stack.min(), stack.max())

    # Load model
    seg_path = find_weights('debris_segmenter')
    log.info("Loading DebrisSegmenter from %s", seg_path)
    model = DebrisSegmenter(patch_size=64)
    model.load_state_dict(torch.load(seg_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Run inference
    log.info("Running sliding window inference...")
    prob_map = sliding_window_inference(
        stack, model,
        patch_size=64,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )
    log.info(
        "Result: shape=%s, range=[%.4f, %.4f], mean=%.4f",
        prob_map.shape, prob_map.min(), prob_map.max(), prob_map.mean(),
    )

    # Build output DataArray with spatial coords from dataset
    prob_da = xr.DataArray(
        prob_map,
        dims=['y', 'x'],
        coords={'y': ds.y, 'x': ds.x},
        name='debris_probability',
    )
    prob_da = prob_da.rio.write_crs(ds.rio.crs)

    # Save
    out_path = args.out or (args.nc.parent / f"scene_debris_{args.date}.tif")
    prob_da.rio.to_raster(str(out_path))
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
