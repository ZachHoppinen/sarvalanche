"""Run pairwise CNN inference on a season dataset and save pair probabilities.

Usage:
    conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
        --nc local/issw/uac/netcdfs/Salt_Lake/season_2023-2024_Salt_Lake.nc \
        --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v3_best.pt \
        --out-dir local/issw/uac/inference/Salt_Lake/

    # Batch: all zones for a center
    conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
        --nc-dir local/issw/uac/netcdfs/ \
        --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v3_best.pt \
        --out-dir local/issw/uac/inference/
"""

import argparse
import gc
import logging
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def build_post_context(ds, pair_meta, anf_norm):
    """Build (2, H, W) post-event VV/VH context channels for a pair.

    Normalizes post-event dB magnitudes to [0, 1] using [-30, 5] dB range.
    """
    DB_MIN, DB_MAX = -30.0, 5.0
    t_end = pair_meta['t_end']
    # Find the global time index closest to t_end
    times = pd.DatetimeIndex(ds.time.values)
    tj = int(np.argmin(np.abs(times - t_end)))

    post_vv = ds['VV'].isel(time=tj).values.astype(np.float32)
    post_vh = ds['VH'].isel(time=tj).values.astype(np.float32) if 'VH' in ds else np.zeros_like(post_vv)

    post_vv = np.clip((np.nan_to_num(post_vv, nan=DB_MIN) - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
    post_vh = np.clip((np.nan_to_num(post_vh, nan=DB_MIN) - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
    return np.stack([post_vv, post_vh], axis=0)


def run_inference_on_nc(nc_path, weights_path, out_dir, device=None,
                        max_span_days=60, stride=32, batch_size=16):
    """Run pairwise CNN inference on a single season nc."""
    from sarvalanche.io.dataset import load_netcdf_to_dataset
    from sarvalanche.ml.pairwise_debris_classifier.inference import (
        load_model, build_sar_channels, sliding_window_inference,
    )
    from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import extract_all_pairs
    from sarvalanche.ml.pairwise_debris_classifier.static_stack import build_static_stack

    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    zone_name = nc_path.stem  # e.g. season_2023-2024_Salt_Lake

    # Output paths
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / f"{zone_name}_pair_probs.npz"
    meta_path = out_dir / f"{zone_name}_pair_meta.csv"

    if probs_path.exists() and meta_path.exists():
        log.info("Skipping %s — inference results already exist", zone_name)
        return

    t0 = time.time()
    log.info("Loading %s", nc_path)
    ds = load_netcdf_to_dataset(nc_path)
    if any(v.chunks is not None for v in ds.variables.values()):
        ds = ds.load()

    # Load model and detect channel count
    model, torch_device = load_model(weights_path, device=device)
    in_ch = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Get actual in_ch from first conv layer
    first_weight = list(model.parameters())[0]
    model_in_ch = first_weight.shape[1]
    use_post_context = model_in_ch > 9
    log.info("Model in_ch=%d, post_context=%s", model_in_ch, use_post_context)

    # Build static stack
    static_scene = build_static_stack(ds)
    log.info("Static stack: %s", static_scene.shape)

    # Extract pairs
    log.info("Extracting pairs (max_span=%d days)...", max_span_days)
    pair_diffs, pair_metas, anf_per_track, _ = extract_all_pairs(
        ds, max_span_days=max_span_days)
    log.info("Total pairs: %d", len(pair_diffs))

    # Run inference per pair
    pair_probs = []
    for pi, (vv_arr, vh_arr, valid_mask) in enumerate(
        tqdm(pair_diffs, desc=f"  Pairs ({zone_name})", leave=True,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    ):
        meta = pair_metas[pi]
        anf_norm = anf_per_track[meta['track']]

        sar = build_sar_channels(vv_arr, vh_arr, anf_norm)

        # Add post-context channels if model expects them
        if use_post_context:
            post = build_post_context(ds, meta, anf_norm)
            sar = np.concatenate([sar, post], axis=0)  # (6, H, W)

        prob_map = sliding_window_inference(
            sar, static_scene, model, torch_device,
            stride=stride, batch_size=batch_size,
            verbose=False,
        )
        prob_map[~valid_mask] = np.nan
        pair_probs.append(prob_map)

    # Save pair probabilities as compressed npz
    np.savez_compressed(probs_path, *pair_probs)
    log.info("Saved %d probability maps to %s", len(pair_probs), probs_path)

    # Save pair metadata as CSV
    meta_df = pd.DataFrame([
        {
            'pair_idx': i,
            'track': m['track'],
            't_start': m['t_start'].isoformat(),
            't_end': m['t_end'].isoformat(),
            'span_days': m['span_days'],
        }
        for i, m in enumerate(pair_metas)
    ])
    meta_df.to_csv(meta_path, index=False)
    log.info("Saved metadata to %s", meta_path)

    elapsed = time.time() - t0
    log.info("Done %s in %.0f min (%.1f s/pair)",
             zone_name, elapsed / 60, elapsed / max(len(pair_probs), 1))

    del ds, pair_probs, pair_diffs
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise CNN inference on season datasets",
    )
    nc_group = parser.add_mutually_exclusive_group(required=True)
    nc_group.add_argument("--nc", type=Path, help="Single season nc")
    nc_group.add_argument("--nc-dir", type=Path,
                          help="Directory of zone subdirs with season ncs")

    parser.add_argument("--weights", type=Path, required=True,
                        help="Model checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for inference results")
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.nc:
        run_inference_on_nc(
            args.nc, args.weights, args.out_dir,
            device=args.device, max_span_days=args.max_span_days,
            stride=args.stride, batch_size=args.batch_size,
        )
    else:
        # Batch mode: find all season ncs
        ncs = sorted(args.nc_dir.rglob("season_*_*.nc"))
        ncs = [nc for nc in ncs if "v2_season" not in str(nc)
               and "v3_" not in str(nc)
               and "probabilities" not in str(nc)]
        log.info("Found %d season datasets", len(ncs))
        for ni, nc in enumerate(ncs, 1):
            log.info("[%d/%d] %s", ni, len(ncs), nc.name)
            zone_dir = args.out_dir / nc.parent.name
            try:
                run_inference_on_nc(
                    nc, args.weights, zone_dir,
                    device=args.device, max_span_days=args.max_span_days,
                    stride=args.stride, batch_size=args.batch_size,
                )
            except Exception as e:
                log.error("Failed %s: %s", nc.name, e)


if __name__ == "__main__":
    main()
