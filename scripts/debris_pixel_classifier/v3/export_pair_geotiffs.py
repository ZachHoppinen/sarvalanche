"""Export dB VV change and model probability as GeoTIFFs for selected pairs.

Usage:
    # Export 2 specific pairs by index (fast — only extracts those pairs):
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/export_pair_geotiffs.py \
        --weights local/issw/snfac/v3_experiment/v3_snfac_best.pt \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --out-dir figures/snfac/v3_pair_geotiffs \
        --pair-index 0 50

    # Export 10 evenly-spaced pairs:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/export_pair_geotiffs.py \
        --weights local/issw/snfac/v3_experiment/v3_snfac_best.pt \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --out-dir figures/snfac/v3_pair_geotiffs \
        --max-pairs 10
"""

import argparse
import logging
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import torch
import xarray as xr
from tqdm import tqdm

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_STATIC
from sarvalanche.ml.v3.model import SinglePairDetector
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_season_pairs,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def sliding_window_inference(sar_scene, static_scene, model, device,
                              patch_size=V3_PATCH_SIZE, stride=32, batch_size=16,
                              model_sar_ch=None):
    """Run sliding window CNN inference (no TTA)."""
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        n_batches = (len(coords) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(coords), batch_size), total=n_batches,
                    desc="  Inference", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for batch_start in pbar:
            batch_coords = coords[batch_start:batch_start + batch_size]
            patches = []
            for y0, x0 in batch_coords:
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                if model_sar_ch is not None and sar_patch.shape[0] > model_sar_ch:
                    sar_patch = sar_patch[:model_sar_ch]
                static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                static_patch = normalize_dem_patch(static_patch)
                patches.append(np.concatenate([sar_patch, static_patch], axis=0))

            batch_tensor = torch.from_numpy(np.stack(patches)).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    return np.where(count > 0, prob_sum / count, np.nan).astype(np.float32)


def save_geotiff(data, ds, out_path, nodata=np.nan):
    """Save a 2D array as a GeoTIFF using the dataset's CRS and coordinates."""
    da = xr.DataArray(
        data, dims=["y", "x"],
        coords={"y": ds.y.values, "x": ds.x.values},
    )
    da = da.rio.set_crs(ds.rio.crs)
    da.rio.to_raster(str(out_path), dtype="float32")
    log.info("  Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=10,
                        help="Number of evenly-spaced pairs to export (ignored if --pair-index given)")
    parser.add_argument("--pair-index", type=int, nargs="+", default=None,
                        help="Specific pair indices to export (0-based)")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-span-days", type=int, default=60)
    args = parser.parse_args()

    t_total = _time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # Load model
    t0 = _time.time()
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        log.info("Checkpoint epoch=%s, val_loss=%.4f", ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    else:
        state_dict = ckpt

    in_ch = state_dict["enc1.block.0.weight"].shape[1]
    base_ch = state_dict["enc1.block.0.weight"].shape[0]
    model = SinglePairDetector(in_ch=in_ch, base_ch=base_ch).to(device)
    model.load_state_dict(state_dict)
    model_sar_ch = in_ch - N_STATIC
    log.info("Model loaded: in_ch=%d (sar=%d, static=%d), base_ch=%d (%.1fs)",
             in_ch, model_sar_ch, N_STATIC, base_ch, _time.time() - t0)

    # Load dataset
    t0 = _time.time()
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("Dataset loaded: %d times, %dx%d (%.1fs)",
             ds.sizes["time"], ds.sizes["y"], ds.sizes["x"], _time.time() - t0)

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR: %d times", hrrr_ds.sizes["time"])

    # Static stack
    t0 = _time.time()
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("Static stack: %s (%.1fs)", static_scene.shape, _time.time() - t0)

    # Get pairs
    t0 = _time.time()
    log.info("Computing pairs (max_span=%dd)...", args.max_span_days)
    pairs = get_all_season_pairs(ds, max_span_days=args.max_span_days, hrrr_ds=hrrr_ds)
    log.info("%d total pairs (%.1fs)", len(pairs), _time.time() - t0)

    # Select pairs
    pairs_sorted = sorted(pairs, key=lambda p: (str(p["t_end"]), p["span_days"]))

    if args.pair_index is not None:
        selected = []
        for idx in args.pair_index:
            if 0 <= idx < len(pairs_sorted):
                selected.append(pairs_sorted[idx])
            else:
                log.warning("Pair index %d out of range (0-%d), skipping", idx, len(pairs_sorted) - 1)
    else:
        step = max(1, len(pairs_sorted) // args.max_pairs)
        selected = pairs_sorted[::step][:args.max_pairs]

    # Print pair list so user can pick indices next time
    log.info("Selected %d pairs for export:", len(selected))
    for i, p in enumerate(selected):
        log.info("  [%d] trk%s %s → %s (%dd)",
                 i, p["track"], str(p["t_start"])[:10], str(p["t_end"])[:10], p["span_days"])

    # Export
    for pi, pair in enumerate(selected):
        t_pair = _time.time()
        t_start = str(pair["t_start"])[:10]
        t_end = str(pair["t_end"])[:10]
        track = pair["track"]
        span = pair["span_days"]
        tag = f"trk{track}_{t_start}_{t_end}_{span}d"
        log.info("[%d/%d] %s — running inference...", pi + 1, len(selected), tag)

        change_vv = pair["sar"][0].copy()

        prob_map = sliding_window_inference(
            pair["sar"], static_scene, model, device,
            stride=args.stride, batch_size=args.batch_size,
            model_sar_ch=model_sar_ch,
        )

        # Mask no-coverage areas
        no_cov = np.abs(pair["sar"][0]) < 1e-6
        prob_map[no_cov] = np.nan
        change_vv[no_cov] = np.nan

        n_above = int(np.nansum(prob_map > 0.2))
        log.info("  prob stats: max=%.3f, >0.2=%d px, >0.5=%d px",
                 float(np.nanmax(prob_map)), n_above, int(np.nansum(prob_map > 0.5)))

        save_geotiff(change_vv, ds, args.out_dir / f"{tag}_change_vv.tif")
        save_geotiff(prob_map, ds, args.out_dir / f"{tag}_prob.tif")
        log.info("  Pair done in %.1fs", _time.time() - t_pair)

    log.info("All done. %d pairs exported to %s (total %.1fs)",
             len(selected), args.out_dir, _time.time() - t_total)


if __name__ == "__main__":
    main()
