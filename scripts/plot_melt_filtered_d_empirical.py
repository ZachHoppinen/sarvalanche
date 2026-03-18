"""Plot d_empirical with and without melt-pair filtering.

For a reference date, computes d_empirical two ways:
1. Standard: all SAR pairs contribute
2. Melt-filtered: pairs where either acquisition had T > 0°C at that pixel are excluded

Shows that removing warm-contaminated pairs reduces false positive signal from melt.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.weights.temporal import get_temporal_weights
from sarvalanche.detection.backscatter_change import (
    calculate_empirical_backscatter_probability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

NC = "local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc"
HRRR = "local/cnfaic/hrrr_temperature_test.nc"
OUT_DIR = "local/cnfaic/figures"
TAU = 6


def compute_d_empirical(ds, date_str, tau=TAU):
    """Standard d_empirical computation."""
    import re
    ref_ts = np.datetime64(pd.Timestamp(date_str))
    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)
    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}
    _, d_emp = calculate_empirical_backscatter_probability(
        ds, ref_ts, use_agreement_boosting=True,
        agreement_strength=0.8, min_prob_threshold=0.2,
    )
    return d_emp.values, ds


def compute_d_empirical_melt_filtered(ds, date_str, hrrr_ds, tau=TAU):
    """Compute d_empirical but mask out SAR timesteps where T > 0°C.

    For each SAR time step, check the HRRR t2m_max on that date.
    If t2m_max > 0°C at a pixel (resampled from 3km), that timestep's
    contribution is zeroed out for that pixel.

    We do this by setting the temporal weight to 0 for warm timesteps,
    then recomputing d_empirical.
    """
    import re
    ref_ts = np.datetime64(pd.Timestamp(date_str))

    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    # Standard temporal weights
    w_temporal = get_temporal_weights(ds["time"], ref_ts, tau_days=tau)

    # For each SAR timestep, check if that date was warm
    sar_times = pd.DatetimeIndex(ds.time.values)
    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    # Build a per-pixel, per-timestep mask: True = warm (exclude)
    # t2m_max is (time, y, x) at 30m resolution (already resampled in HRRR nc)
    warm_mask_per_step = []
    n_masked = 0
    for i, sar_t in enumerate(sar_times):
        sar_date = sar_t.strftime("%Y-%m-%d")
        # Find closest HRRR date
        time_diffs = np.abs(hrrr_times - sar_t)
        ci = time_diffs.argmin()
        if time_diffs[ci].days > 2:
            # No HRRR data near this date — keep it
            warm_mask_per_step.append(np.zeros(ds["slope"].shape, dtype=bool))
            continue
        t2m_max = hrrr_ds["t2m_max"].isel(time=ci).values
        is_warm = t2m_max > 0.0  # above freezing at this pixel
        warm_mask_per_step.append(is_warm)
        if is_warm.any():
            frac = is_warm.sum() / is_warm.size
            log.info("  %s: %.1f%% of pixels warm (T>0°C)", sar_date, frac * 100)
            n_masked += 1

    log.info("  %d / %d SAR timesteps have warm pixels", n_masked, len(sar_times))

    # Build per-pixel 3D weights: start from temporal weights, zero warm timesteps
    w_vals = w_temporal.values  # (time,)
    w_3d = np.broadcast_to(w_vals[:, None, None], (len(sar_times),) + ds["slope"].shape).copy()

    # Diagnostic: which timesteps survive (cold) before and after ref date?
    ref_ts = pd.Timestamp(date_str)
    for i, sar_t in enumerate(sar_times):
        is_before = sar_t < ref_ts
        sar_date = sar_t.strftime("%Y-%m-%d")
        warm_frac = warm_mask_per_step[i].sum() / warm_mask_per_step[i].size
        w = w_vals[i]
        if w > 0.01:
            tag = "BEFORE" if is_before else "AFTER"
            status = f"WARM({warm_frac*100:.0f}%)" if warm_frac > 0.01 else "cold"
            log.info("    %s %s  w=%.3f  %s", tag, sar_date, w, status)

    # Zero out weights for warm timesteps at each pixel
    for i, warm in enumerate(warm_mask_per_step):
        w_3d[i, warm] = 0.0

    # Renormalize: before and after groups should each sum to 1
    # so cold pairs with longer baselines get upweighted to compensate
    before_mask = sar_times < ref_ts
    after_mask = sar_times >= ref_ts

    w_before_3d = w_3d.copy()
    w_before_3d[~before_mask] = 0
    w_after_3d = w_3d.copy()
    w_after_3d[~after_mask] = 0

    # Renormalize each group per-pixel
    sum_before = w_before_3d.sum(axis=0, keepdims=True)
    sum_after = w_after_3d.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        w_before_3d = np.where(sum_before > 0, w_before_3d / sum_before, 0)
        w_after_3d = np.where(sum_after > 0, w_after_3d / sum_after, 0)

    result_d = np.zeros(ds["slope"].shape, dtype=np.float64)
    result_n = np.zeros(ds["slope"].shape, dtype=np.float64)

    for pol in ["VV", "VH"]:
        if pol not in ds:
            continue
        sar = ds[pol].values  # (time, y, x), linear scale
        sar_db = np.where(sar > 0, 10 * np.log10(sar), np.nan)

        with np.errstate(invalid="ignore"):
            mean_before = np.nansum(sar_db * w_before_3d, axis=0)
            mean_after = np.nansum(sar_db * w_after_3d, axis=0)

        # Only valid where both before and after have contributing data
        has_before = sum_before.squeeze() > 0
        has_after = sum_after.squeeze() > 0
        change = mean_after - mean_before  # signed: positive = brighter after
        valid = np.isfinite(change) & has_before & has_after
        result_d[valid] += change[valid]
        result_n[valid] += 1

    with np.errstate(invalid="ignore"):
        d_filtered = np.where(result_n > 0, result_d / result_n, np.nan)

    return d_filtered


def compute_d_empirical_melt_filtered_fallback(d_standard, d_filtered):
    """Fill NaN gaps in melt-filtered result with standard d_empirical.

    When all pairs at a pixel are warm, we have no cold pairs to use.
    Fall back to the standard value rather than leaving holes.
    """
    result = d_filtered.copy()
    gaps = np.isnan(result) & np.isfinite(d_standard)
    result[gaps] = d_standard[gaps]
    return result


def plot_comparison(d_standard, d_filtered, ds, date_str, out_path):
    """Side-by-side plot: standard vs melt-filtered d_empirical."""
    slope = ds["slope"].values
    valid = (slope > 0.09) & (slope < 1.05)

    # Stats
    s_std = d_standard[valid & np.isfinite(d_standard)]
    s_filt = d_filtered[valid & np.isfinite(d_filtered)]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Independent color scales per panel so both are readable
    p98_std = np.nanpercentile(s_std, 98) if len(s_std) > 0 else 6
    p98_filt = np.nanpercentile(s_filt, 98) if len(s_filt) > 0 else 2
    vmax_std = max(p98_std, 2.0)
    vmax_filt = max(p98_filt, 0.5)

    im0 = axes[0].imshow(np.clip(d_standard, 0, vmax_std),
                          cmap="hot_r", vmin=0, vmax=vmax_std,
                          extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()])
    axes[0].set_title(f"Standard d_empirical\n"
                      f"mean={np.nanmean(s_std):.2f} dB, "
                      f">3dB: {(s_std > 3).sum():,} px ({100*(s_std>3).sum()/len(s_std):.1f}%)")

    im1 = axes[1].imshow(np.clip(d_filtered, 0, vmax_filt),
                          cmap="hot_r", vmin=0, vmax=vmax_filt,
                          extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()])
    axes[1].set_title(f"Melt-filtered d_empirical (scale 0–{vmax_filt:.1f} dB)\n"
                      f"mean={np.nanmean(s_filt):.2f} dB, "
                      f">3dB: {(s_filt > 3).sum():,} px ({100*(s_filt>3).sum()/len(s_filt):.1f}%)")

    # Difference
    diff = d_standard - d_filtered
    diff_valid = diff[valid & np.isfinite(diff)]
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-2, vmax=2,
                          extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()])
    axes[2].set_title(f"Difference (standard − filtered)\n"
                      f"mean={np.nanmean(diff_valid):.2f} dB, "
                      f"red = signal removed by filtering")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.colorbar(im0, ax=axes[0], label="dB", shrink=0.7)
    plt.colorbar(im1, ax=axes[1], label="dB", shrink=0.7)
    plt.colorbar(im2, ax=axes[2], label="dB difference", shrink=0.7)

    fig.suptitle(f"d_empirical: {date_str} — Effect of removing melt-contaminated pairs (T>0°C)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def main():
    from pathlib import Path

    log.info("Loading season dataset...")
    ds = load_netcdf_to_dataset(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    log.info("Loading HRRR temperature...")
    hrrr_ds = xr.open_dataset(HRRR)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dates to compare: ones with warm pairs reaching in
    dates = ["2025-12-15", "2026-02-13", "2026-02-25"]

    for date_str in dates:
        log.info("=== %s ===", date_str)

        log.info("Computing standard d_empirical...")
        d_std, ds = compute_d_empirical(ds, date_str)

        log.info("Computing melt-filtered d_empirical...")
        d_filt_raw = compute_d_empirical_melt_filtered(ds, date_str, hrrr_ds)

        # Fill gaps where all pairs were warm — fall back to standard
        d_filt = compute_d_empirical_melt_filtered_fallback(d_std, d_filt_raw)
        n_fallback = np.isnan(d_filt_raw) & np.isfinite(d_std)
        log.info("  Fallback to standard for %d pixels (%.1f%% — all pairs warm)",
                 n_fallback.sum(), 100 * n_fallback.sum() / np.isfinite(d_std).sum())

        out_path = out_dir / f"melt_filter_{date_str}.png"
        plot_comparison(d_std, d_filt, ds, date_str, str(out_path))

    hrrr_ds.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
