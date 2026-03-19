"""Run v3 single-pair inference on one date and visualize.

For a given reference date, extracts all crossing pairs, runs the CNN
on each pair independently, and saves per-pair probability GeoTIFFs.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/infer_single_pair.py \
        --nc local/cnfaic/netcdfs/.../season_2025-2026_*.nc \
        --weights local/cnfaic/v3_experiment/weights/v3_best.pt \
        --date 2025-12-15 \
        --hrrr local/cnfaic/hrrr_temperature_2526.nc \
        --out-dir local/cnfaic/v3_experiment/inference_pairs/2025-12-15
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import torch
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT, N_SAR
from sarvalanche.ml.v3.model import SinglePairDetector
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_pairs,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def sliding_window_inference(sar_scene, static_scene, model, device, patch_size=V3_PATCH_SIZE,
                              stride=32, batch_size=16):
    """Run CNN on sliding windows, average overlapping predictions.

    Parameters
    ----------
    sar_scene : (N_SAR, H, W) one pair
    static_scene : (N_STATIC, H, W)

    Returns
    -------
    prob_map : (H, W) float32 probability
    """
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    # Collect patch coordinates
    coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((y0, x0))

    # Batch inference
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            patches = []

            for y0, x0 in batch_coords:
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                static_patch = normalize_dem_patch(static_patch)

                x = np.concatenate([sar_patch, static_patch], axis=0)
                patches.append(x)

            batch_tensor = torch.from_numpy(np.stack(patches)).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    prob_map = np.where(count > 0, prob_sum / count, 0.0).astype(np.float32)
    return prob_map


def compute_empirical_for_date(ds, date_str, tau=6):
    """Compute empirical layers needed for static stack."""
    import re
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(pd.Timestamp(date_str))
    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_emp, d_emp = calculate_empirical_backscatter_probability(
        ds, ref_ts, use_agreement_boosting=True,
        agreement_strength=0.8, min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_emp
    ds["d_empirical"] = d_emp
    return ds


def main():
    parser = argparse.ArgumentParser(description="v3 single-pair inference + visualization")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info("Device: %s", device)

    # Load model
    model = SinglePairDetector(in_ch=N_INPUT, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    log.info("Loaded weights: %s", args.weights)

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        hrrr_ds = xr.open_dataset(args.hrrr)

    # Compute empirical for static stack
    log.info("Computing empirical for %s...", args.date)
    ds = compute_empirical_for_date(ds, args.date)

    # Build static stack
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)

    # Get all crossing pairs
    log.info("Getting crossing pairs...")
    pairs = get_all_pairs(ds, args.date, max_pairs_per_track=args.max_pairs, hrrr_ds=hrrr_ds)
    log.info("  %d pairs", len(pairs))

    # Run inference on each pair
    crs = ds.rio.crs
    x_coords, y_coords = ds.x.values, ds.y.values

    pair_probs = []
    for pi, pair in enumerate(pairs):
        ts = str(pair['t_start'])[:10]
        te = str(pair['t_end'])[:10]
        mw = pair['melt_weight_mean']
        log.info("  [%d/%d] %s → %s (track %s, span %dd, melt_w=%.2f)",
                 pi + 1, len(pairs), ts, te, pair['track'], pair['span_days'], mw)

        prob_map = sliding_window_inference(
            pair['sar'], static_scene, model, device,
            stride=args.stride, batch_size=args.batch_size,
        )

        n50 = (prob_map > 0.5).sum()
        n20 = (prob_map > 0.2).sum()
        log.info("    >0.5: %d px, >0.2: %d px, mean: %.4f", n50, n20, prob_map.mean())

        # Save GeoTIFF
        da = xr.DataArray(prob_map, dims=['y', 'x'],
                          coords={'y': y_coords, 'x': x_coords})
        da = da.rio.write_crs(crs)
        fname = f"pair_{pi:02d}_{ts}_{te}_track{pair['track']}.tif"
        da.rio.to_raster(str(args.out_dir / fname))

        pair_probs.append({
            'prob_map': prob_map,
            'proximity': 1.0 / (1.0 + pair['span_days'] / 12.0),
            'melt_weight': mw,
        })

    # Weighted mean across pairs
    if pair_probs:
        w_total = np.zeros_like(pair_probs[0]['prob_map'])
        p_total = np.zeros_like(pair_probs[0]['prob_map'])
        for pp in pair_probs:
            w = pp['proximity'] * pp['melt_weight']
            p_total += pp['prob_map'] * w
            w_total += w
        weighted_mean = np.where(w_total > 0, p_total / w_total, 0.0).astype(np.float32)

        da = xr.DataArray(weighted_mean, dims=['y', 'x'],
                          coords={'y': y_coords, 'x': x_coords})
        da = da.rio.write_crs(crs)
        da.rio.to_raster(str(args.out_dir / f"weighted_mean_{args.date}.tif"))

        n50 = (weighted_mean > 0.5).sum()
        n20 = (weighted_mean > 0.2).sum()
        log.info("Weighted mean: >0.5: %d px, >0.2: %d px", n50, n20)

    if hrrr_ds is not None:
        hrrr_ds.close()
    log.info("Saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
