"""Single-pair patch extraction for v3 debris detector.

Extracts one (N_SAR + N_STATIC, 128, 128) patch per crossing pair.
Each pair is an independent training/inference sample.

v3 differences from v2:
  - No set encoder: each pair is evaluated independently
  - 7 SAR channels per pair (change, ANF, proximity, melt_weight,
    VV mag, VH mag, cross-ratio)
  - 13 static channels (adds curvature, TPI; drops raw d_empirical)
  - Validation path masking: patches overlapping AKDOT/AKRR paths are
    held out for validation
"""

import logging
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from sarvalanche.ml.v3.channels import (
    N_SAR,
    N_STATIC,
    SAR_CHANNELS,
    STATIC_CHANNELS,
    normalize_sar_channel,
    normalize_static_channel,
)

log = logging.getLogger(__name__)

V3_PATCH_SIZE = 128


# ── SAR pair extraction ──────────────────────────────────────────────

def _normalize_anf(arr: np.ndarray) -> np.ndarray:
    """Normalize ANF: 1 = best quality (low ANF)."""
    return 1.0 / (1.0 + np.log1p(arr))


def _hrrr_pdd_melt_weight(hrrr_ds, sar_date, hrrr_times, pdd_threshold=0.1):
    """Compute per-pixel melt weight from HRRR. 0=warm, 1=cold."""
    time_diffs = np.abs(hrrr_times - sar_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 2:
        return None

    if 'pdd_24h' in hrrr_ds and 't2m_max' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        t2m = hrrr_ds['t2m_max'].isel(time=ci).values
        pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
        t2m_smooth = gaussian_filter(t2m, sigma=15, mode='nearest')
        pdd_weight = np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0)
        t2m_weight = np.clip((-t2m_smooth - 3.0) / 5.0, 0.0, 1.0)
        return np.minimum(pdd_weight, t2m_weight).astype(np.float32)
    elif 'pdd_24h' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
        return np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0).astype(np.float32)
    return None


def extract_single_pair(
    da_vv, da_vh, i, j, times,
    anf_norm, hrrr_cache, tau_proximity=12.0,
):
    """Extract 7-channel SAR array for one crossing pair.

    Parameters
    ----------
    da_vv, da_vh : (time, y, x) dB arrays for one track
    i, j : time indices (before, after)
    times : DatetimeIndex
    anf_norm : (y, x) normalized ANF for this track
    hrrr_cache : dict mapping Timestamp → melt_weight array or None
    tau_proximity : float

    Returns
    -------
    (N_SAR, H, W) float32 array, or None if data is missing
    """
    H, W = da_vv.shape[1], da_vv.shape[2]
    span_days = (times[j] - times[i]).days

    # VV change
    vv_before = da_vv[i]
    vv_after = da_vv[j]
    vv_diff = (vv_after - vv_before).astype(np.float32)
    vv_diff = np.nan_to_num(vv_diff, nan=0.0)

    # Skip if mostly NaN (no SAR coverage)
    if np.isfinite(vv_after).sum() < 100:
        return None

    # log1p compression
    change = np.sign(vv_diff) * np.log1p(np.abs(vv_diff))

    # Temporal proximity
    proximity = np.full((H, W), 1.0 / (1.0 + span_days / tau_proximity), dtype=np.float32)

    # Melt weight: min of both endpoints
    melt_w = np.ones((H, W), dtype=np.float32)
    mw_i = hrrr_cache.get(times[i])
    mw_j = hrrr_cache.get(times[j])
    if mw_i is not None:
        melt_w = np.minimum(melt_w, mw_i)
    if mw_j is not None:
        melt_w = np.minimum(melt_w, mw_j)

    # VV and VH magnitudes (mean of before and after, in dB)
    vv_mag = np.nan_to_num((vv_before + vv_after) / 2.0, nan=-25.0).astype(np.float32)
    vv_mag_norm = normalize_sar_channel(vv_mag, 'vv_magnitude')

    if da_vh is not None:
        vh_before = da_vh[i]
        vh_after = da_vh[j]
        vh_mag = np.nan_to_num((vh_before + vh_after) / 2.0, nan=-25.0).astype(np.float32)
        vh_mag_norm = normalize_sar_channel(vh_mag, 'vh_magnitude')
        # Per-pair cross-ratio: VH - VV in dB
        cr = np.nan_to_num(vh_mag - vv_mag, nan=0.0).astype(np.float32)
        cr_norm = normalize_sar_channel(cr, 'cross_ratio')
    else:
        vh_mag_norm = np.zeros((H, W), dtype=np.float32)
        cr_norm = np.zeros((H, W), dtype=np.float32)

    # Stack: change, anf, proximity, melt_weight, vv_mag, vh_mag, cross_ratio
    stacked = np.stack([
        change, anf_norm, proximity, melt_w,
        vv_mag_norm, vh_mag_norm, cr_norm,
    ], axis=0)

    return stacked.astype(np.float32)


def get_all_pairs(
    ds: xr.Dataset,
    reference_date,
    max_pairs_per_track: int = 4,
    tau_proximity: float = 12.0,
    hrrr_ds: xr.Dataset | None = None,
) -> list[dict]:
    """Extract all crossing pairs as independent samples.

    Returns list of dicts with:
      'sar': (N_SAR, H, W) float32
      'track': str
      'pol': str (always 'VV' — VH is folded into magnitude/CR channels)
      't_start': Timestamp
      't_end': Timestamp
      'span_days': int
      'melt_weight_mean': float (scene-average melt trust for this pair)
    """
    from sarvalanche.utils.generators import iter_track_pol_combinations
    from sarvalanche.preprocessing.radiometric import linear_to_dB
    from sarvalanche.utils.validation import check_db_linear

    ref = pd.Timestamp(reference_date)
    has_anf = 'anf' in ds.data_vars

    # Precompute HRRR melt weights
    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        for t in pd.DatetimeIndex(ds.time.values):
            hrrr_cache[t] = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)

    # Group track/pol: we want VV as primary, VH for magnitude/CR
    # Iterate by track, grab both VV and VH
    tracks_seen = set()
    results = []

    for track, pol, da in iter_track_pol_combinations(ds):
        if pol != 'VV':
            continue  # we handle VH inside when we have VV
        if track in tracks_seen:
            continue
        tracks_seen.add(track)

        # Get VV in dB
        if check_db_linear(da) != 'dB':
            da = linear_to_dB(da)
        vv_vals = da.values  # (time, y, x) dB

        # Get VH if available
        vh_vals = None
        for t2, p2, da2 in iter_track_pol_combinations(ds):
            if t2 == track and p2 == 'VH':
                if check_db_linear(da2) != 'dB':
                    da2 = linear_to_dB(da2)
                vh_vals = da2.values
                break

        times = pd.DatetimeIndex(da.time.values)

        # ANF for this track
        if has_anf:
            track_int = int(track)
            if track_int in ds['anf'].static_track.values:
                anf_arr = np.nan_to_num(
                    ds['anf'].sel(static_track=track_int).values.astype(np.float32),
                    nan=1.0,
                )
                anf_norm = _normalize_anf(anf_arr)
            else:
                anf_norm = np.ones((ds.sizes['y'], ds.sizes['x']), dtype=np.float32)
        else:
            anf_norm = np.ones((ds.sizes['y'], ds.sizes['x']), dtype=np.float32)

        # Find crossing pairs
        pairs = []
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[i] <= ref < times[j]:
                    pairs.append((i, j, (times[j] - times[i]).days))
        if not pairs:
            continue

        # Sort by tightest span, take top K
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:max_pairs_per_track]

        for i, j, span in pairs:
            sar = extract_single_pair(
                vv_vals, vh_vals, i, j, times,
                anf_norm, hrrr_cache, tau_proximity,
            )
            if sar is None:
                continue

            melt_w_mean = float(sar[SAR_CHANNELS.index('melt_weight')].mean())
            results.append({
                'sar': sar,
                'track': str(track),
                't_start': times[i],
                't_end': times[j],
                'span_days': span,
                'melt_weight_mean': melt_w_mean,
            })

    log.info('get_all_pairs: %d pairs for date %s', len(results), ref.date())
    return results


# ── Static stack ─────────────────────────────────────────────────────

DEM_CHANNEL: int = STATIC_CHANNELS.index('dem')
_PATCH_NORM_CHANNELS: set[str] = {'dem'}


def build_static_stack(
    ds: xr.Dataset,
    hrrr_ds: xr.Dataset | None = None,
) -> np.ndarray:
    """Build (N_STATIC, H, W) static terrain stack.

    Includes curvature, TPI, and melt-filtered d_empirical.
    """
    H, W = ds.sizes['y'], ds.sizes['x']
    stack = np.zeros((N_STATIC, H, W), dtype=np.float32)

    # Aspect decomposition
    aspect_derived = {}
    if 'aspect' in ds.data_vars:
        aspect = np.nan_to_num(ds['aspect'].values.astype(np.float32), nan=0.0)
        aspect_derived['aspect_northing'] = np.cos(aspect)
        aspect_derived['aspect_easting'] = np.sin(aspect)

    # Compute curvature and TPI from DEM
    derived = {}
    if 'dem' in ds.data_vars:
        dem = np.nan_to_num(ds['dem'].values.astype(np.float32), nan=0.0)
        try:
            from sarvalanche.utils.terrain import compute_curvature, compute_tpi
            # Need projected CRS for terrain metrics
            if ds.rio.crs and ds.rio.crs.is_geographic:
                # Approximate pixel size in meters
                mid_lat = float(ds.y.values.mean())
                dx_m = abs(float(ds.x.values[1] - ds.x.values[0])) * 111320 * np.cos(np.radians(mid_lat))
            else:
                dx_m = abs(float(ds.x.values[1] - ds.x.values[0]))

            derived['curvature'] = compute_curvature(dem, dx_m)
            derived['tpi'] = compute_tpi(dem, radius_px=5)
            log.info('  Computed curvature and TPI from DEM')
        except Exception as e:
            log.warning('  Could not compute curvature/TPI: %s', e)
            derived['curvature'] = np.zeros((H, W), dtype=np.float32)
            derived['tpi'] = np.zeros((H, W), dtype=np.float32)

    # Cross-ratio change (pooled across tracks)
    if 'd_cr' in STATIC_CHANNELS:
        from sarvalanche.ml.v2.patch_extraction import _compute_d_cr
        d_cr = _compute_d_cr(ds, H, W)
        if d_cr is not None:
            derived['d_cr'] = d_cr

    # Melt-filtered d_empirical
    if hrrr_ds is not None and 'd_empirical_melt_filtered' in STATIC_CHANNELS:
        from sarvalanche.ml.v2.patch_extraction import _compute_melt_filtered_d_empirical
        d_filt = _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W)
        if d_filt is not None:
            derived['d_empirical_melt_filtered'] = d_filt

    # Fill static stack
    for ch, var in enumerate(STATIC_CHANNELS):
        if var in aspect_derived:
            stack[ch] = aspect_derived[var]
        elif var in derived:
            arr = derived[var]
            if var not in _PATCH_NORM_CHANNELS:
                arr = normalize_static_channel(arr, var)
            stack[ch] = arr
        elif var in ds.data_vars:
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            if var not in _PATCH_NORM_CHANNELS:
                arr = normalize_static_channel(arr, var)
            stack[ch] = arr

    return stack


def normalize_dem_patch(static_patch: np.ndarray) -> np.ndarray:
    """Per-patch min-max normalize the DEM channel."""
    dem = static_patch[DEM_CHANNEL]
    dem_min = dem.min()
    dem_range = dem.max() - dem_min
    if dem_range > 1e-6:
        static_patch[DEM_CHANNEL] = (dem - dem_min) / dem_range
    else:
        static_patch[DEM_CHANNEL] = 0.0
    return static_patch
