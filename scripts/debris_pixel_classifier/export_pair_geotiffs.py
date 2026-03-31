"""Export dB VV change and model probability as GeoTIFFs for selected pairs.

Uses the pairwise_debris_classifier module (memmap if available).

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/export_pair_geotiffs.py \
        --weights src/sarvalanche/ml/weights/pairwise_debris_detector/combined_best.pt \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --out-dir figures/snfac/pairwise_geotiffs \
        --max-pairs 2
"""

import argparse
import logging
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from sarvalanche.ml.pairwise_debris_classifier.inference import (
    load_model, build_sar_channels, sliding_window_inference,
)
from sarvalanche.ml.pairwise_debris_classifier.static_stack import build_static_stack
from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import (
    get_track_data, extract_all_pairs,
)
from sarvalanche.io.dataset import load_netcdf_to_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def save_geotiff(data, ds, out_path):
    da = xr.DataArray(data, dims=["y", "x"],
                       coords={"y": ds.y.values, "x": ds.x.values})
    da = da.rio.set_crs(ds.rio.crs)
    da.rio.to_raster(str(out_path), dtype="float32")
    log.info("  Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--pair-index", type=int, nargs="+", default=None)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-span-days", type=int, default=30,
                        help="Max pair span in days (shorter = more likely to show debris)")
    args = parser.parse_args()

    t_total = _time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, device = load_model(args.weights, device=args.device)

    # Load dataset
    t0 = _time.time()
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("Loaded in %.0fs (%d times, %dx%d)",
             _time.time() - t0, ds.sizes["time"], ds.sizes["y"], ds.sizes["x"])

    # Static stack
    t0 = _time.time()
    static_scene = build_static_stack(ds)
    log.info("Static stack: %s (%.0fs)", static_scene.shape, _time.time() - t0)

    # Extract pairs
    t0 = _time.time()
    pair_diffs, pair_metas, anf_per_track, _ = extract_all_pairs(
        ds, max_span_days=args.max_span_days)
    log.info("%d pairs (%.0fs)", len(pair_diffs), _time.time() - t0)

    # Select pairs
    if args.pair_index is not None:
        selected = [(i, pair_diffs[i], pair_metas[i]) for i in args.pair_index
                     if i < len(pair_diffs)]
    else:
        step = max(1, len(pair_diffs) // args.max_pairs)
        selected = [(i, pair_diffs[i], pair_metas[i])
                     for i in range(0, len(pair_diffs), step)][:args.max_pairs]

    log.info("Exporting %d pairs:", len(selected))
    for i, _, meta in selected:
        log.info("  [%d] trk%s %s -> %s (%dd)",
                 i, meta['track'], str(meta['t_start'])[:10],
                 str(meta['t_end'])[:10], meta['span_days'])

    # Export
    for pi, (idx, (vv_arr, vh_arr, valid_mask), meta) in enumerate(selected):
        t_pair = _time.time()
        t_start = str(meta['t_start'])[:10]
        t_end = str(meta['t_end'])[:10]
        track = meta['track']
        span = meta['span_days']
        tag = f"trk{track}_{t_start}_{t_end}_{span}d"
        log.info("[%d/%d] %s", pi + 1, len(selected), tag)

        anf_norm = anf_per_track[track]
        sar = build_sar_channels(vv_arr, vh_arr, anf_norm)

        prob_map = sliding_window_inference(
            sar, static_scene, model, device,
            stride=args.stride, batch_size=args.batch_size,
        )
        prob_map[~valid_mask] = np.nan

        # Raw dB VV change (not log1p transformed — raw diff for visualization)
        change_vv_raw = vv_arr.copy()
        change_vv_raw[~valid_mask] = np.nan

        n_above = int(np.nansum(prob_map > 0.2))
        log.info("  prob: max=%.3f, >0.2=%d px, >0.5=%d px",
                 float(np.nanmax(prob_map)), n_above, int(np.nansum(prob_map > 0.5)))

        save_geotiff(change_vv_raw, ds, args.out_dir / f"{tag}_change_vv_dB.tif")
        save_geotiff(prob_map, ds, args.out_dir / f"{tag}_prob.tif")
        log.info("  Done in %.1fs", _time.time() - t_pair)

    log.info("All done. %d pairs in %.0fs", len(selected), _time.time() - t_total)


if __name__ == "__main__":
    main()
