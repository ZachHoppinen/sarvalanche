"""Run v3 single-pair inference for a full season.

Computes ALL unique pairs once (span <= max_span_days), runs CNN on each,
saves per-pair probability GeoTIFFs. These can be reused for any reference
date via the temporal onset detector.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/infer_season.py \
        --nc local/cnfaic/netcdfs/.../season_2025-2026_*.nc \
        --weights local/cnfaic/v3_experiment/weights/v3_best.pt \
        --hrrr local/cnfaic/hrrr_temperature_2526.nc \
        --out-dir local/cnfaic/v3_experiment/season_pairs
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import torch
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT
from sarvalanche.ml.v3.model import SinglePairDetector
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_season_pairs,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


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


def sliding_window_inference(sar_scene, static_scene, model, device,
                              patch_size=V3_PATCH_SIZE, stride=32, batch_size=16):
    """Run CNN on sliding windows, average overlapping predictions."""
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            patches = []
            for y0, x0 in batch_coords:
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                static_patch = normalize_dem_patch(static_patch)
                patches.append(np.concatenate([sar_patch, static_patch], axis=0))

            batch_tensor = torch.from_numpy(np.stack(patches)).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    return np.where(count > 0, prob_sum / count, 0.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="v3 full-season pair inference")
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
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
    log.info("Loaded weights: %s (%d params)", args.weights,
             sum(p.numel() for p in model.parameters()))

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
        log.info("Loaded HRRR from %s", args.hrrr)

    # We need empirical for the static stack (d_empirical_melt_filtered, d_cr).
    # Use the season midpoint as a reference — the static stack is recomputed
    # per-date if needed, but for curvature/TPI/terrain it doesn't matter.
    times = pd.DatetimeIndex(ds.time.values)
    in_season = times[(times.month >= 11) | (times.month <= 4)]
    mid_date = in_season[len(in_season) // 2]
    log.info("Computing empirical for static stack (ref=%s)...", mid_date.date())
    ds = compute_empirical_for_date(ds, str(mid_date)[:10])

    # Build static stack (shared across all pairs for a given date)
    # For simplicity, build one static stack. The d_empirical_melt_filtered
    # channel is date-dependent, but for season inference we update it per-date.
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)

    # Get ALL unique pairs in the season
    log.info("Extracting all season pairs (max span %dd)...", args.max_span_days)
    pairs = get_all_season_pairs(ds, max_span_days=args.max_span_days, hrrr_ds=hrrr_ds)
    log.info("  %d unique pairs to process", len(pairs))

    # Save pair metadata
    pair_meta = []
    crs = ds.rio.crs
    x_coords, y_coords = ds.x.values, ds.y.values

    for pi, pair in enumerate(pairs):
        ts = str(pair['t_start'])[:10]
        te = str(pair['t_end'])[:10]
        mw = pair['melt_weight_mean']

        log.info("  [%d/%d] %s → %s (track %s, %dd, melt_w=%.2f)",
                 pi + 1, len(pairs), ts, te, pair['track'], pair['span_days'], mw)

        prob_map = sliding_window_inference(
            pair['sar'], static_scene, model, device,
            stride=args.stride, batch_size=args.batch_size,
        )

        n50 = int((prob_map > 0.5).sum())
        n20 = int((prob_map > 0.2).sum())
        log.info("    >0.5: %d px, >0.2: %d px, mean: %.4f", n50, n20, prob_map.mean())

        # Save GeoTIFF
        fname = f"pair_{pi:03d}_{ts}_{te}_track{pair['track']}.tif"
        da = xr.DataArray(prob_map, dims=['y', 'x'],
                          coords={'y': y_coords, 'x': x_coords})
        da = da.rio.write_crs(crs)
        da.rio.to_raster(str(args.out_dir / fname))

        pair_meta.append({
            'pair_idx': pi,
            'track': pair['track'],
            't_start': ts,
            't_end': te,
            'span_days': pair['span_days'],
            'melt_weight_mean': round(mw, 3),
            'n_above_50': n50,
            'n_above_20': n20,
            'mean_prob': round(float(prob_map.mean()), 5),
            'filename': fname,
        })

    # Save metadata
    with open(args.out_dir / 'pair_metadata.json', 'w') as f:
        json.dump(pair_meta, f, indent=2)

    log.info("Done. %d pair GeoTIFFs saved to %s", len(pairs), args.out_dir)
    log.info("Next: run temporal_onset.py on subsets of these pairs per reference date")

    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
