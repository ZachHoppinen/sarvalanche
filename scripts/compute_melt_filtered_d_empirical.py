"""Compute d_empirical with per-pair melt filtering using HRRR temperature.

Replicates the exact pipeline from the sarvalanche detection system:
  1. For each track/pol: compute all crossing pairs (backscatter_changes_crossing_date)
  2. Weight pairs by temporal proximity (get_temporal_weights)
  3. Weighted mean of pair diffs → d per track/pol
  4. Combine across track/pol with resolution + polarization weights → d_empirical

The melt filtering modification:
  For each pair (t_start, t_end), look up HRRR t2m_max at both dates.
  At pixels where either date had T > 0°C, that pair's weight is zeroed.
  Remaining weights are renormalized per-pixel so cold pairs fill in.

Usage:
    conda run -n sarvalanche python scripts/compute_melt_filtered_d_empirical.py \
        --nc local/cnfaic/netcdfs/.../season_2025-2026_*.nc \
        --hrrr local/cnfaic/hrrr_temperature_test.nc \
        --date 2025-12-15 \
        --out-dir local/cnfaic/figures
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.probabilities.features import probability_backscatter_change
from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.weights.combinations import combine_weights, weighted_mean
from sarvalanche.weights.polarizations import get_polarization_weights
from sarvalanche.weights.temporal import get_temporal_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

TAU = 6


def _hrrr_melt_weight_for_date(hrrr_ds, sar_date, hrrr_times):
    """Look up HRRR 24h PDD for a SAR date. Returns 2D float (y, x): 0=warm, 1=cold.

    Uses cumulative positive degree-day (PDD) over 24 hours, which smooths
    temporal noise and captures both the intensity and duration of warmth.

    Continuous ramp:
        PDD = 0      → 1.0  (no positive temps in 24h, fully trusted)
        PDD = 0.5    → 0.5  (half a degree-day, partial)
        PDD >= 1.0   → 0.0  (significant melt energy, fully suppressed)

    Falls back to t2m_max-based weight if PDD not available.
    """
    from scipy.ndimage import gaussian_filter

    time_diffs = np.abs(hrrr_times - sar_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 2:
        return None  # no HRRR data near this date

    if "pdd_24h" in hrrr_ds:
        pdd = hrrr_ds["pdd_24h"].isel(time=ci).values  # degree-days

        # Smooth to avoid blocky 3km edges
        pdd_smooth = gaussian_filter(pdd, sigma=15, mode="nearest")

        # Very aggressive ramp: PDD=0 → 1.0, PDD≥0.1 → 0.0
        # Any meaningful positive temperature exposure suppresses
        melt_weight = np.clip(1.0 - pdd_smooth / 0.1, 0.0, 1.0)
    else:
        # Fallback to t2m_max if PDD not available
        t2m = hrrr_ds["t2m_max"].isel(time=ci).values
        t2m_smooth = gaussian_filter(t2m, sigma=15, mode="nearest")
        melt_weight = np.clip((-t2m_smooth - 1.0) / 4.0, 0.0, 1.0)

    return melt_weight


def compute_track_d_empirical_melt_filtered(da, avalanche_date, hrrr_ds, hrrr_times, tau=TAU):
    """Compute d_empirical for a single track/pol with per-pair per-pixel melt filtering.

    Exactly replicates compute_track_empirical_probability but with melt masking.

    Returns (d_filtered, d_standard) both as (y, x) numpy arrays, or None if no pairs.
    """
    # Convert to dB
    if check_db_linear(da) != "dB":
        da = linear_to_dB(da)

    # Get all crossing pairs (same as pipeline)
    try:
        diffs = backscatter_changes_crossing_date(da, avalanche_date)
    except ValueError:
        return None

    # Standard temporal weights (same as pipeline)
    w_pair = get_temporal_weights(diffs["t_start"], diffs["t_end"], tau_days=tau)

    # Standard d: weighted mean of pair diffs (nansum to handle missing SAR coverage)
    diffs_np = diffs.values  # (pair, y, x)
    w_1d = w_pair.values  # (pair,)
    w_std_3d = np.broadcast_to(w_1d[:, None, None], diffs_np.shape).copy()
    # Zero weight where diff is NaN, then renormalize
    nan_mask = np.isnan(diffs_np)
    w_std_3d[nan_mask] = 0.0
    w_std_sum = w_std_3d.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        w_std_3d = np.where(w_std_sum > 0, w_std_3d / w_std_sum, 0.0)
    d_standard = np.nansum(diffs_np * w_std_3d, axis=0)
    d_standard[w_std_sum.squeeze() == 0] = np.nan

    # --- Melt filtering ---
    # For each pair, build a per-pixel warm mask from HRRR at t_start and t_end
    n_pairs = len(diffs["pair"])
    spatial_shape = diffs.isel(pair=0).shape  # (y, x)
    w_3d = np.broadcast_to(
        w_pair.values[:, None, None], (n_pairs,) + spatial_shape
    ).copy()

    n_pairs_downweighted = 0
    for p in range(n_pairs):
        t_start = pd.Timestamp(diffs["t_start"].values[p])
        t_end = pd.Timestamp(diffs["t_end"].values[p])

        mw_start = _hrrr_melt_weight_for_date(hrrr_ds, t_start, hrrr_times)
        mw_end = _hrrr_melt_weight_for_date(hrrr_ds, t_end, hrrr_times)

        # Pair trust = min of both endpoints (worst case drives the penalty)
        pair_melt_weight = np.ones(spatial_shape, dtype=np.float64)
        if mw_start is not None:
            pair_melt_weight = np.minimum(pair_melt_weight, mw_start)
        if mw_end is not None:
            pair_melt_weight = np.minimum(pair_melt_weight, mw_end)

        # Multiply temporal weight by melt trust
        w_3d[p] *= pair_melt_weight
        if (pair_melt_weight < 0.99).any():
            n_pairs_downweighted += 1

    # Also zero weights where diffs are NaN (missing SAR coverage)
    w_3d[nan_mask] = 0.0

    # Renormalize weights per-pixel so surviving pairs sum to 1
    w_sum = w_3d.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        w_3d_norm = np.where(w_sum > 0, w_3d / w_sum, 0.0)

    # Weighted mean of pair diffs with filtered weights
    d_filtered = np.nansum(diffs_np * w_3d_norm, axis=0)

    # Where no pairs survived, fall back to standard
    no_data = w_sum.squeeze() == 0
    d_filtered[no_data] = d_standard[no_data]

    log.info(
        "    %d/%d pairs downweighted by melt, fallback at %d px (%.2f%%)",
        n_pairs_downweighted, n_pairs, no_data.sum(),
        100 * no_data.sum() / no_data.size,
    )

    return d_filtered, d_standard


def compute_d_empirical_melt_filtered(ds, date_str, hrrr_ds, tau=TAU):
    """Full d_empirical computation with melt filtering, replicating the real pipeline.

    Returns (d_filtered, d_standard) as (y, x) arrays.
    """
    avalanche_date = np.datetime64(pd.Timestamp(date_str))
    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    # Clean stale vars
    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    # Set temporal weights (needed by pipeline internals)
    ds["w_temporal"] = get_temporal_weights(ds["time"], avalanche_date, tau_days=tau)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    # Process each track/pol (same as calculate_empirical_backscatter_probability)
    filtered_results = []
    standard_results = []
    resolution_weights = []
    pol_weights = []
    labels = []

    for track, pol, da in iter_track_pol_combinations(ds):
        log.info("  track=%s pol=%s", track, pol)
        result = compute_track_d_empirical_melt_filtered(
            da, avalanche_date, hrrr_ds, hrrr_times, tau=tau,
        )
        if result is None:
            log.warning("    Skipped (no crossing pairs)")
            continue

        d_filt, d_std = result
        filtered_results.append(d_filt)
        standard_results.append(d_std)
        resolution_weights.append(ds["w_resolution"].sel(static_track=track).values)
        pol_weights.append(get_polarization_weights(pol))
        labels.append(f"{track}_{pol}")

    if not filtered_results:
        raise ValueError("No track/pol results")

    # Combine across track/pol with resolution + polarization weights
    # (same as pipeline: combine_weights then weighted_mean)
    res_w = np.array([r.mean() if hasattr(r, "mean") else r for r in resolution_weights])
    pol_w = np.array(pol_weights)
    combined_w = res_w * pol_w
    combined_w = combined_w / combined_w.sum()

    log.info("Combining %d track/pol combos: %s", len(labels), labels)
    for i, lbl in enumerate(labels):
        log.info("  %s: weight=%.3f", lbl, combined_w[i])

    # Weighted average across track/pol — per-pixel renormalization
    # Each track only covers part of the grid, so we must redistribute
    # weights at each pixel based on which tracks have data there.
    spatial_shape = filtered_results[0].shape
    n_combos = len(labels)

    # Stack into (n_combos, y, x)
    filt_stack = np.stack(filtered_results, axis=0)
    std_stack = np.stack(standard_results, axis=0)

    # Per-pixel weight: zero where track has no data (NaN)
    w_3d = np.broadcast_to(combined_w[:, None, None], (n_combos,) + spatial_shape).copy()
    w_3d[np.isnan(filt_stack)] = 0.0
    w_sum = w_3d.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        w_3d_norm = np.where(w_sum > 0, w_3d / w_sum, 0.0)

    d_filt_combined = np.nansum(filt_stack * w_3d_norm, axis=0)
    d_filt_combined[w_sum.squeeze() == 0] = np.nan

    # Same for standard
    w_3d_std = np.broadcast_to(combined_w[:, None, None], (n_combos,) + spatial_shape).copy()
    w_3d_std[np.isnan(std_stack)] = 0.0
    w_sum_std = w_3d_std.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        w_3d_std_norm = np.where(w_sum_std > 0, w_3d_std / w_sum_std, 0.0)

    d_std_combined = np.nansum(std_stack * w_3d_std_norm, axis=0)
    d_std_combined[w_sum_std.squeeze() == 0] = np.nan

    has_any = w_sum.squeeze() > 0
    log.info("Spatial coverage: %d px (%.1f%% of grid)", has_any.sum(), 100 * has_any.mean())

    return d_filt_combined, d_std_combined, ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("local/cnfaic/figures"))
    parser.add_argument("--tau", type=float, default=6.0, help="Temporal decay constant in days")
    args = parser.parse_args()

    log.info("Loading dataset...")
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    log.info("Loading HRRR...")
    hrrr_ds = xr.open_dataset(args.hrrr)

    log.info("Computing d_empirical for %s (tau=%.0f days)...", args.date, args.tau)
    d_filt, d_std, ds = compute_d_empirical_melt_filtered(ds, args.date, hrrr_ds, tau=args.tau)

    # Stats
    valid = np.isfinite(d_filt)
    v_filt = d_filt[valid]
    v_std = d_std[valid]
    log.info("Standard:      min=%.2f max=%.2f mean=%.2f  >3dB: %d px",
             v_std.min(), v_std.max(), v_std.mean(), (v_std > 3).sum())
    log.info("Melt-filtered: min=%.2f max=%.2f mean=%.2f  >3dB: %d px",
             v_filt.min(), v_filt.max(), v_filt.mean(), (v_filt > 3).sum())

    # Save GeoTIFFs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = f"tau{int(args.tau)}"
    for name, arr in [(f"d_empirical_standard_{tau_tag}", d_std),
                      (f"d_empirical_melt_filtered_{tau_tag}", d_filt)]:
        da = xr.DataArray(arr.astype(np.float32), dims=["y", "x"],
                          coords={"y": ds.y.values, "x": ds.x.values})
        da = da.rio.write_crs(ds.rio.crs)
        out = args.out_dir / f"{name}_{args.date}.tif"
        da.rio.to_raster(str(out))
        log.info("Saved %s", out)

    hrrr_ds.close()


if __name__ == "__main__":
    main()
