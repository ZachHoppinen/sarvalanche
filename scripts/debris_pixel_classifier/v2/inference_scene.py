"""Whole-scene sliding window inference for v2 debris detector.

Slides a 128x128 window across the scene at configurable stride, runs the
model on per-track/pol SAR maps + static terrain, and averages overlapping
predictions into a full-scene probability map.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/inference_scene.py \
        --nc local/issw/dual_tau_output/zone/season_dataset.nc \
        --date 2025-02-04 \
        --tau 6 \
        --weights v2_detector_best.pt \
        --out scene_v2_prob.tif
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import torch
import xarray as xr
from tqdm.auto import tqdm

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v2.model import DebrisDetector
from sarvalanche.ml.v2.patch_extraction import (
    V2_PATCH_SIZE,
    build_static_stack,
    get_per_track_pol_changes,
    normalize_dem_patch,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def compute_empirical_for_date(ds, reference_date, tau_days):
    """Compute per-track/pol empirical layers for a given date."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)

    stale_patterns = [
        re.compile(r'^p_\d+_V[VH]_empirical$'),
        re.compile(r'^d_\d+_V[VH]_empirical$'),
    ]
    stale_exact = {'p_empirical', 'd_empirical', 'w_temporal'}
    to_drop = [v for v in ds.data_vars if v in stale_exact]
    for pat in stale_patterns:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        ds = ds.drop_vars(to_drop)

    ds['w_temporal'] = get_temporal_weights(ds['time'], ref_ts, tau_days=tau_days)
    ds['w_temporal'].attrs = {'source': 'sarvalanche', 'units': '1', 'product': 'weight'}

    p_empirical, d_empirical = calculate_empirical_backscatter_probability(
        ds, ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )
    ds['p_empirical'] = p_empirical
    ds['d_empirical'] = d_empirical
    return ds


def sliding_window_inference(
    sar_change_maps: list[np.ndarray],
    static_stack: np.ndarray,
    model: DebrisDetector,
    patch_size: int = V2_PATCH_SIZE,
    stride: int = 32,
    batch_size: int = 16,
    device: str = 'cpu',
) -> np.ndarray:
    """Run v2 model over full scene with sliding window.

    Parameters
    ----------
    sar_change_maps : list of (H, W) arrays, one per track/pol
    static_stack : (N_STATIC, H, W)
    model : DebrisDetector
    patch_size, stride, batch_size, device : inference settings

    Returns:
    -------
    (H, W) probability map averaged over overlapping patches.
    """
    H, W = sar_change_maps[0].shape
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    # Collect patch coordinates
    coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((y0, x0))

    # Handle edges
    if (H - patch_size) % stride != 0:
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((H - patch_size, x0))
    if (W - patch_size) % stride != 0:
        for y0 in range(0, H - patch_size + 1, stride):
            coords.append((y0, W - patch_size))
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        coords.append((H - patch_size, W - patch_size))
    coords = list(dict.fromkeys(coords))

    log.info(
        'Sliding window: %d patches (%dx%d scene, patch=%d, stride=%d, %d track/pols)',
        len(coords), H, W, patch_size, stride, len(sar_change_maps),
    )

    model.eval()

    n_batches = (len(coords) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), batch_size), total=n_batches, desc='Inference'):
            batch_coords = coords[i:i + batch_size]

            # Build batch SAR maps: list of (B, 1, H, W)
            sar_batch = []
            for change_map in sar_change_maps:
                patches = np.stack([
                    change_map[y0:y0 + patch_size, x0:x0 + patch_size]
                    for y0, x0 in batch_coords
                ])
                sar_batch.append(
                    torch.from_numpy(patches[:, np.newaxis]).float().to(device)
                )

            # Static batch: (B, C, H, W) — per-patch DEM normalization
            static_patches = np.stack([
                normalize_dem_patch(
                    static_stack[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                )
                for y0, x0 in batch_coords
            ])
            static_t = torch.from_numpy(static_patches).float().to(device)

            logits = model(sar_batch, static_t)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for j, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[j]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    mask = count > 0
    result = np.zeros((H, W), dtype=np.float32)
    result[mask] = (prob_sum[mask] / count[mask]).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run v2 debris detector over full scene'
    )
    parser.add_argument('--nc', type=Path, required=True, help='Path to season_dataset.nc')
    parser.add_argument('--date', type=str, required=True, help='Reference date (YYYY-MM-DD)')
    parser.add_argument('--tau', type=float, default=6, help='Temporal decay tau in days')
    parser.add_argument('--weights', type=Path, required=True, help='Path to model weights .pt')
    parser.add_argument('--out', type=Path, default=None, help='Output GeoTIFF path')
    parser.add_argument('--stride', type=int, default=32, help='Sliding window stride (default: 32)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detect if omitted)')
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
    log.info('Using device: %s', device)

    # Load dataset
    log.info('Loading dataset: %s', args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds['time'].dtype, np.datetime64):
        ds['time'] = pd.DatetimeIndex(ds['time'].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info('  %d time steps, %dx%d spatial', len(ds.time), ds.sizes['y'], ds.sizes['x'])

    # Ensure w_resolution
    if 'w_resolution' not in ds.data_vars:
        from sarvalanche.weights.local_resolution import get_local_resolution_weights
        log.info('Computing resolution weights...')
        ds['w_resolution'] = get_local_resolution_weights(ds['anf'])
        ds['w_resolution'].attrs = {'source': 'sarvalanche', 'units': '1', 'product': 'weight'}

    # Compute empirical layers (produces per-track/pol d_*_empirical)
    ref_date = pd.Timestamp(args.date)
    log.info('Computing empirical layers for %s (tau=%gd)...', args.date, args.tau)
    ds = compute_empirical_for_date(ds, ref_date, args.tau)

    # Build inputs
    track_pol_maps = get_per_track_pol_changes(ds)
    if not track_pol_maps:
        log.error('No per-track/pol change maps found. Check dataset.')
        return
    log.info('Found %d track/pol change maps', len(track_pol_maps))

    sar_change_maps = [arr for _, _, arr in track_pol_maps]
    static_stack = build_static_stack(ds)
    log.info('Static stack shape: %s', static_stack.shape)

    # Load model
    log.info('Loading model from %s', args.weights)
    model = DebrisDetector()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)

    # Run inference
    log.info('Running sliding window inference...')
    prob_map = sliding_window_inference(
        sar_change_maps, static_stack, model,
        patch_size=V2_PATCH_SIZE,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )
    log.info(
        'Result: shape=%s, range=[%.4f, %.4f], mean=%.4f',
        prob_map.shape, prob_map.min(), prob_map.max(), prob_map.mean(),
    )

    # Build output DataArray
    prob_da = xr.DataArray(
        prob_map,
        dims=['y', 'x'],
        coords={'y': ds.y, 'x': ds.x},
        name='debris_probability_v2',
    )
    prob_da = prob_da.rio.write_crs(ds.rio.crs)

    out_path = args.out or (args.nc.parent / f'scene_v2_debris_{args.date}.tif')
    prob_da.rio.to_raster(str(out_path))
    log.info('Saved to %s', out_path)


if __name__ == '__main__':
    main()
