"""Full-season CNN inference: run DebrisDetector at every SAR acquisition date.

Loads the dataset, model, and static terrain stack once, then for each in-season
date recomputes only the temporal-dependent layers (empirical backscatter) and
runs sliding-window inference. Outputs a time-series NetCDF with dims
(time, y, x) containing per-date debris probability maps, plus individual
GeoTIFFs per date.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/full_season_inference.py \
        --nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
        --weights local/issw/v2_patches/v2_detector_best.pt \
        --season 2024-2025 \
        --tau 6
"""

import argparse
import gc
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import torch
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v2.channels import N_STATIC, STATIC_CHANNELS, normalize_static_channel
from sarvalanche.ml.v2.model import DebrisDetector, DebrisDetectorSkip
from sarvalanche.ml.v2.patch_extraction import (
    V2_PATCH_SIZE,
    DEM_CHANNEL,
    _PATCH_NORM_CHANNELS,
    normalize_dem_patch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stale variable patterns
# ---------------------------------------------------------------------------

_STALE_PATTERNS = [
    re.compile(r"^p_\d+_V[VH]_empirical$"),
    re.compile(r"^d_\d+_V[VH]_empirical$"),
    re.compile(r"^m_\d+_V[VH]_empirical$"),
]
_STALE_EXACT = {"p_empirical", "d_empirical", "w_temporal"}


def _clean_stale(ds):
    """Drop temporal-dependent variables between steps."""
    to_drop = [v for v in ds.data_vars if v in _STALE_EXACT]
    for pat in _STALE_PATTERNS:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        ds = ds.drop_vars(to_drop)
    return ds


def compute_empirical(ds, reference_date, tau_days):
    """Compute empirical layers for one date. Returns ds or None on failure."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    try:
        p_emp, d_emp = calculate_empirical_backscatter_probability(
            ds, ref_ts,
            use_agreement_boosting=True,
            agreement_strength=0.8,
            min_prob_threshold=0.2,
            tau_days=tau_days,
        )
    except ValueError as e:
        log.warning("Empirical failed for %s: %s", reference_date, e)
        return None

    ds["p_empirical"] = p_emp
    ds["d_empirical"] = d_emp
    return ds


# ---------------------------------------------------------------------------
# Static stack — build once, update d_empirical channel per step
# ---------------------------------------------------------------------------

def build_static_stack_base(ds):
    """Build static stack with all channels except date-dependent ones.

    Returns (stack, d_empirical_channel_index, d_cr_channel_index_or_None).
    """
    H, W = ds.sizes["y"], ds.sizes["x"]
    stack = np.zeros((N_STATIC, H, W), dtype=np.float32)

    aspect_derived = {}
    if "aspect" in ds.data_vars and (
        "aspect_northing" in STATIC_CHANNELS or "aspect_easting" in STATIC_CHANNELS
    ):
        aspect = np.nan_to_num(ds["aspect"].values.astype(np.float32), nan=0.0)
        aspect_derived["aspect_northing"] = np.cos(aspect)
        aspect_derived["aspect_easting"] = np.sin(aspect)

    d_emp_idx = STATIC_CHANNELS.index("d_empirical")
    d_cr_idx = STATIC_CHANNELS.index("d_cr") if "d_cr" in STATIC_CHANNELS else None
    # Skip date-dependent channels (filled per-date)
    skip_vars = {"d_empirical", "d_cr", "d_empirical_melt_filtered", "d_empirical_melt_residual"}

    for ch, var in enumerate(STATIC_CHANNELS):
        if var in skip_vars:
            continue
        if var in aspect_derived:
            stack[ch] = aspect_derived[var]
        elif var in ds.data_vars:
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            if var not in _PATCH_NORM_CHANNELS:
                arr = normalize_static_channel(arr, var)
            stack[ch] = arr

    return stack, d_emp_idx, d_cr_idx


def update_empirical_channel(static_stack, ds, d_emp_idx, hrrr_ds=None):
    """Update d_empirical and melt channels in-place."""
    # Standard d_empirical always stays as-is
    arr = np.nan_to_num(ds["d_empirical"].values.astype(np.float32), nan=0.0)
    static_stack[d_emp_idx] = normalize_static_channel(arr, "d_empirical")

    # Melt-filtered + residual channels if HRRR available
    if hrrr_ds is not None:
        from sarvalanche.ml.v2.channels import STATIC_CHANNELS
        if 'd_empirical_melt_filtered' in STATIC_CHANNELS:
            from sarvalanche.ml.v2.patch_extraction import _compute_melt_filtered_d_empirical
            H, W = ds.sizes["y"], ds.sizes["x"]
            d_filtered = _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W)
            if d_filtered is not None:
                filt_idx = STATIC_CHANNELS.index('d_empirical_melt_filtered')
                resid_idx = STATIC_CHANNELS.index('d_empirical_melt_residual')
                static_stack[filt_idx] = normalize_static_channel(d_filtered, "d_empirical_melt_filtered")
                residual = arr - d_filtered
                static_stack[resid_idx] = normalize_static_channel(residual, "d_empirical_melt_residual")


def update_d_cr_channel(static_stack, ds, d_cr_idx):
    """Update the cross-ratio change channel in-place."""
    if d_cr_idx is None:
        return
    from sarvalanche.ml.v2.patch_extraction import _compute_d_cr
    H, W = ds.sizes["y"], ds.sizes["x"]
    d_cr = _compute_d_cr(ds, H, W)
    if d_cr is not None:
        static_stack[d_cr_idx] = normalize_static_channel(d_cr, "d_cr")
    else:
        static_stack[d_cr_idx] = 0.0


# ---------------------------------------------------------------------------
# SAR map extraction
# ---------------------------------------------------------------------------

def get_sar_change_maps(ds):
    """Extract per-track/pol change maps from current empirical variables.

    Returns list of (2, H, W) arrays:
        ch 0: log1p-compressed backscatter change
        ch 1: normalized ANF
    """
    from sarvalanche.ml.v2.patch_extraction import normalize_anf

    pattern = re.compile(r"^d_(\d+)_(V[VH])_empirical$")
    has_anf = "anf" in ds.data_vars
    results = []
    for var in sorted(ds.data_vars):
        m = pattern.match(var)
        if m:
            track = m.group(1)
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            arr = np.sign(arr) * np.log1p(np.abs(arr))

            if has_anf:
                anf_arr = np.nan_to_num(
                    ds["anf"].sel(static_track=int(track)).values.astype(np.float32),
                    nan=1.0,
                )
                anf_norm = normalize_anf(anf_arr)
            else:
                anf_norm = np.ones_like(arr)

            results.append(np.stack([arr, anf_norm], axis=0))
    return results


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def build_patch_coords(H, W, patch_size, stride):
    """Pre-compute sliding window coordinates (done once)."""
    coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((y0, x0))
    if (H - patch_size) % stride != 0:
        for x0 in range(0, W - patch_size + 1, stride):
            coords.append((H - patch_size, x0))
    if (W - patch_size) % stride != 0:
        for y0 in range(0, H - patch_size + 1, stride):
            coords.append((y0, W - patch_size))
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        coords.append((H - patch_size, W - patch_size))
    return list(dict.fromkeys(coords))


def run_inference(sar_maps, static_stack, model, coords, patch_size, batch_size, device, H, W):
    """Run sliding window inference for one date. Returns (H, W) prob map."""
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]

            sar_batch = []
            for change_map in sar_maps:
                patches = np.stack([
                    change_map[:, y0:y0 + patch_size, x0:x0 + patch_size]
                    for y0, x0 in batch_coords
                ])
                sar_batch.append(
                    torch.from_numpy(patches).float().to(device)
                )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-season CNN inference at every SAR acquisition date",
    )
    parser.add_argument("--nc", type=Path, required=True, help="Path to season_dataset.nc")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights .pt")
    parser.add_argument("--season", type=str, default="2024-2025", help='Season "YYYY-YYYY"')
    parser.add_argument("--tau", type=float, default=6, help="Temporal decay tau (default: 6)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride (default: 32)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--no-tiffs", action="store_true", help="Skip per-date GeoTIFFs, only save NetCDF")
    parser.add_argument("--skip", action="store_true", help="Use DebrisDetectorSkip (skip connections)")
    parser.add_argument("--pairs", action="store_true",
                        help="Use per-pair mode (v2.1): individual crossing pairs instead of pooled change")
    parser.add_argument("--combo", action="store_true",
                        help="Use combo mode: both pooled + per-pair maps (3-ch each)")
    parser.add_argument("--max-pairs", type=int, default=4,
                        help="Max crossing pairs per track/pol in --pairs/--combo mode (default: 4)")
    parser.add_argument("--sar-channels", type=int, default=None,
                        help="SAR input channels (auto-detect: 2 for pooled, 3 for pairs, 4 for pairs+hrrr)")
    parser.add_argument("--hrrr", type=Path, default=None,
                        help="HRRR temperature NetCDF for melt filtering (adds 4th SAR channel)")
    args = parser.parse_args()

    # Load HRRR if provided
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        import xarray as _xr
        hrrr_ds = _xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR temperature from %s", args.hrrr)

    # Auto-detect SAR channels
    if args.sar_channels is None:
        if hrrr_ds is not None and (args.pairs or args.combo):
            args.sar_channels = 4
        elif args.pairs or args.combo:
            args.sar_channels = 3
        else:
            args.sar_channels = 2

    # Parse season
    try:
        start_year, end_year = args.season.split("-")
        season_start = pd.Timestamp(f"{start_year}-11-01")
        season_end = pd.Timestamp(f"{end_year}-04-30")
    except ValueError:
        parser.error('--season must be "YYYY-YYYY"')

    # Device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    # Output directory
    out_dir = args.out_dir or (args.nc.parent / "v2_season_inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset (once) ──────────────────────────────────────────────
    log.info("Loading dataset: %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    H, W = ds.sizes["y"], ds.sizes["x"]
    log.info("  %d time steps, %d×%d spatial", len(ds.time), H, W)

    if "w_resolution" not in ds.data_vars:
        from sarvalanche.weights.local_resolution import get_local_resolution_weights
        log.info("Computing resolution weights...")
        ds["w_resolution"] = get_local_resolution_weights(ds["anf"])
        ds["w_resolution"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    # ── Determine step dates (every unique SAR date in season) ───────────
    all_times = pd.DatetimeIndex(ds["time"].values)
    step_dates = sorted(set(
        all_times[(all_times >= season_start) & (all_times <= season_end)].date
    ))
    log.info(
        "Season %s to %s: %d unique SAR dates to process",
        season_start.date(), season_end.date(), len(step_dates),
    )
    if not step_dates:
        log.error("No dates in season window.")
        return

    # ── Load model (once) ────────────────────────────────────────────────
    log.info("Loading model from %s", args.weights)
    model_cls = DebrisDetectorSkip if args.skip else DebrisDetector
    model = model_cls(sar_in_ch=args.sar_channels)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # ── Build static stack base (once) ───────────────────────────────────
    log.info("Building static terrain stack (d_empirical updated per step)...")
    static_stack, d_emp_idx, d_cr_idx = build_static_stack_base(ds)

    # ── Pre-compute patch coordinates (once) ─────────────────────────────
    coords = build_patch_coords(H, W, V2_PATCH_SIZE, args.stride)
    log.info("  %d sliding window patches per date", len(coords))

    # ── Allocate output array (time, y, x) ───────────────────────────────
    # Pre-allocate full array so we don't hold individual maps in a dict
    prob_cube = np.zeros((len(step_dates), H, W), dtype=np.float32)
    valid_mask = np.zeros(len(step_dates), dtype=bool)
    crs = ds.rio.crs

    # ── Step loop ────────────────────────────────────────────────────────
    for i, date in enumerate(step_dates):
        date_str = str(date)
        log.info("[%d/%d] %s", i + 1, len(step_dates), date_str)

        ds = _clean_stale(ds)

        ds_result = compute_empirical(ds, pd.Timestamp(date), args.tau)
        if ds_result is None:
            log.warning("  Skipping %s", date_str)
            continue
        ds = ds_result

        update_empirical_channel(static_stack, ds, d_emp_idx, hrrr_ds=hrrr_ds)
        update_d_cr_channel(static_stack, ds, d_cr_idx)

        if args.combo:
            from sarvalanche.ml.v2.patch_extraction import (
                get_per_pair_changes,
                get_per_track_pol_changes,
            )
            # Pooled maps padded to 3-ch
            pooled = get_per_track_pol_changes(ds, n_channels=2)
            pooled_3ch = []
            for _t, _p, arr_2ch in pooled:
                prox = np.ones((1, H, W), dtype=np.float32)
                pooled_3ch.append(np.concatenate([arr_2ch, prox], axis=0))
            pair_arrays = get_per_pair_changes(
                ds, pd.Timestamp(date), max_pairs=args.max_pairs,
            )
            sar_maps = pooled_3ch + pair_arrays
        elif args.pairs:
            from sarvalanche.ml.v2.patch_extraction import get_per_pair_changes
            pair_arrays = get_per_pair_changes(
                ds, pd.Timestamp(date), max_pairs=args.max_pairs,
                hrrr_ds=hrrr_ds,
            )
            sar_maps = pair_arrays  # list of (C, H, W), C=3 or 4
        else:
            sar_maps = get_sar_change_maps(ds)
        if not sar_maps:
            log.warning("  Skipping %s — no SAR change maps", date_str)
            continue

        prob_map = run_inference(
            sar_maps, static_stack, model, coords,
            V2_PATCH_SIZE, args.batch_size, device, H, W,
        )

        prob_cube[i] = prob_map
        valid_mask[i] = True

        log.info(
            "  range=[%.4f, %.4f]  mean=%.4f  >0.5: %d px",
            prob_map.min(), prob_map.max(), prob_map.mean(),
            int((prob_map > 0.5).sum()),
        )

        # Save per-date GeoTIFF
        if not args.no_tiffs:
            out_path = out_dir / f"scene_v2_debris_{date_str}.tif"
            prob_da = xr.DataArray(
                prob_map, dims=["y", "x"],
                coords={"y": ds.y, "x": ds.x},
                name="debris_probability_v2",
            )
            prob_da = prob_da.rio.write_crs(crs)
            prob_da.rio.to_raster(str(out_path))

        del sar_maps, prob_map
        gc.collect()

    # ── Save time-series NetCDF ──────────────────────────────────────────
    valid_dates = [step_dates[i] for i in range(len(step_dates)) if valid_mask[i]]
    valid_probs = prob_cube[valid_mask]

    if len(valid_dates) == 0:
        log.error("No dates produced valid results.")
        return

    time_coords = pd.DatetimeIndex(valid_dates)
    prob_ds = xr.Dataset(
        {"debris_probability": (["time", "y", "x"], valid_probs)},
        coords={
            "time": time_coords,
            "y": ds.y,
            "x": ds.x,
        },
    )
    prob_ds["debris_probability"].attrs = {
        "units": "1",
        "long_name": "CNN debris detection probability",
        "tau_days": args.tau,
    }
    prob_ds = prob_ds.rio.write_crs(crs)

    nc_path = out_dir / "season_v2_debris_probabilities.nc"
    prob_ds.to_netcdf(nc_path)
    log.info("Saved time-series NetCDF: %s (%d dates, %.1f MB)",
             nc_path, len(valid_dates), nc_path.stat().st_size / 1e6)

    # ── Summary stats ────────────────────────────────────────────────────
    log.info("")
    log.info("=== Season summary ===")
    log.info("Dates processed: %d / %d", len(valid_dates), len(step_dates))
    log.info("Time range: %s to %s", valid_dates[0], valid_dates[-1])

    # Per-date stats
    for t in range(len(valid_dates)):
        p = valid_probs[t]
        log.info(
            "  %s  mean=%.4f  max=%.4f  >0.3: %5d px  >0.5: %5d px",
            valid_dates[t], p.mean(), p.max(),
            int((p > 0.3).sum()), int((p > 0.5).sum()),
        )


if __name__ == "__main__":
    main()
