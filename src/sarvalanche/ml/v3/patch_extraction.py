"""Single-pair patch extraction for v3 debris detector.

Extracts one (N_SAR + N_STATIC, 128, 128) patch per crossing pair.
Each pair is an independent training/inference sample.

v3 differences from v2:
  - No set encoder: each pair is evaluated independently
  - 3 SAR channels per pair (change, ANF, proximity)
  - 6 static channels (slope, aspect_N/E, DEM, cell_counts, TPI)
  - All pairs with span <= max_span_days (no pair count limit)
  - get_all_season_pairs: compute all unique pairs once, reuse for any date
  - Validation path masking for held-out evaluation
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
    anf_norm, hrrr_cache=None, tau_proximity=12.0,
    tv_weight=1.0,
):
    """Extract 7-channel SAR array for one crossing pair.

    Channels: change_vv, change_vh, change_cr, ANF, proximity,
              change_vv_smooth, change_vh_smooth.

    Returns (N_SAR, H, W) float32 array, or None if data is missing.
    """
    from skimage.restoration import denoise_tv_chambolle

    H, W = da_vv.shape[1], da_vv.shape[2]
    span_days = (times[j] - times[i]).days

    vv_before = da_vv[i]
    vv_after = da_vv[j]
    vv_diff = (vv_after - vv_before).astype(np.float32)
    vv_diff = np.nan_to_num(vv_diff, nan=0.0)

    if np.isfinite(vv_after).sum() < 100:
        return None

    change_vv = np.sign(vv_diff) * np.log1p(np.abs(vv_diff))

    # VH change
    if da_vh is not None:
        vh_before = da_vh[i]
        vh_after = da_vh[j]
        vh_diff = (vh_after - vh_before).astype(np.float32)
        vh_diff = np.nan_to_num(vh_diff, nan=0.0)
        change_vh = np.sign(vh_diff) * np.log1p(np.abs(vh_diff))
    else:
        vh_diff = np.zeros((H, W), dtype=np.float32)
        change_vh = np.zeros((H, W), dtype=np.float32)

    # Cross-ratio change: (VH-VV)_after - (VH-VV)_before = VH_diff - VV_diff
    if da_vh is not None:
        cr_diff = vh_diff - vv_diff
        change_cr = np.sign(cr_diff) * np.log1p(np.abs(cr_diff))
    else:
        change_cr = np.zeros((H, W), dtype=np.float32)

    proximity = np.full((H, W), 1.0 / (1.0 + span_days / tau_proximity), dtype=np.float32)

    # TV-smoothed channels
    vv_smooth = denoise_tv_chambolle(vv_diff, weight=tv_weight).astype(np.float32)
    vh_smooth = denoise_tv_chambolle(vh_diff, weight=tv_weight).astype(np.float32)
    change_vv_s = np.sign(vv_smooth) * np.log1p(np.abs(vv_smooth))
    change_vh_s = np.sign(vh_smooth) * np.log1p(np.abs(vh_smooth))

    # Coverage mask: 1 where real SAR data, 0 where NaN-filled
    coverage = (np.abs(vv_diff) > 1e-6).astype(np.float32)

    stacked = np.stack([change_vv, change_vh, change_cr, anf_norm, proximity,
                        change_vv_s, change_vh_s, coverage], axis=0)
    return stacked.astype(np.float32)


# ── Track data helper ────────────────────────────────────────────────

def _get_track_data(ds):
    """Get VV/VH arrays, ANF, and times per track."""
    from sarvalanche.utils.generators import iter_track_pol_combinations
    from sarvalanche.preprocessing.radiometric import linear_to_dB
    from sarvalanche.utils.validation import check_db_linear

    has_anf = 'anf' in ds.data_vars
    tracks = {}

    for track, pol, da in iter_track_pol_combinations(ds):
        if pol != 'VV':
            continue
        if str(track) in tracks:
            continue

        if check_db_linear(da) != 'dB':
            da = linear_to_dB(da)
        vv_vals = da.values

        vh_vals = None
        for t2, p2, da2 in iter_track_pol_combinations(ds):
            if t2 == track and p2 == 'VH':
                if check_db_linear(da2) != 'dB':
                    da2 = linear_to_dB(da2)
                vh_vals = da2.values
                break

        times = pd.DatetimeIndex(da.time.values)

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

        tracks[str(track)] = {
            'vv': vv_vals, 'vh': vh_vals, 'times': times, 'anf': anf_norm,
        }

    return tracks


# ── Pair extraction (date-specific and full-season) ──────────────────

def get_all_pairs(
    ds: xr.Dataset,
    reference_date,
    max_span_days: int = 60,
    tau_proximity: float = 12.0,
    hrrr_ds: xr.Dataset | None = None,
) -> list[dict]:
    """Extract all crossing pairs for a reference date (span <= max_span_days).

    No limit on pair count — returns every pair within the span cap.
    """
    ref = pd.Timestamp(reference_date)

    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        for t in pd.DatetimeIndex(ds.time.values):
            hrrr_cache[t] = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)

    tracks = _get_track_data(ds)
    results = []

    for track_id, td in tracks.items():
        times = td['times']
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                span = (times[j] - times[i]).days
                if not (times[i] <= ref < times[j]):
                    continue
                if span > max_span_days:
                    continue

                sar = extract_single_pair(
                    td['vv'], td['vh'], i, j, times,
                    td['anf'], hrrr_cache, tau_proximity,
                )
                if sar is None:
                    continue

                # Compute melt weight from HRRR (not a SAR channel anymore,
                # but still needed for temporal aggregation weighting)
                mw_i = hrrr_cache.get(times[i])
                mw_j = hrrr_cache.get(times[j])
                melt_w = 1.0
                if mw_i is not None and mw_j is not None:
                    melt_w = float(np.minimum(mw_i, mw_j).mean())
                elif mw_i is not None:
                    melt_w = float(mw_i.mean())
                elif mw_j is not None:
                    melt_w = float(mw_j.mean())

                results.append({
                    'sar': sar,
                    'track': track_id,
                    't_start': times[i],
                    't_end': times[j],
                    'span_days': span,
                    'melt_weight_mean': melt_w,
                })

    log.info('get_all_pairs: %d pairs for date %s (max span %dd)',
             len(results), ref.date(), max_span_days)
    return results


def get_all_season_pairs(
    ds: xr.Dataset,
    max_span_days: int = 60,
    tau_proximity: float = 12.0,
    hrrr_ds: xr.Dataset | None = None,
) -> list[dict]:
    """Extract ALL unique pairs in the season (not tied to a reference date).

    Compute once, reuse for any date: for temporal onset, select pairs
    where t_start <= ref < t_end.
    """
    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        for t in pd.DatetimeIndex(ds.time.values):
            hrrr_cache[t] = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)

    tracks = _get_track_data(ds)
    results = []
    seen = set()

    for track_id, td in tracks.items():
        times = td['times']
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                span = (times[j] - times[i]).days
                if span > max_span_days:
                    continue

                key = (track_id, str(times[i]), str(times[j]))
                if key in seen:
                    continue
                seen.add(key)

                sar = extract_single_pair(
                    td['vv'], td['vh'], i, j, times,
                    td['anf'], hrrr_cache, tau_proximity,
                )
                if sar is None:
                    continue

                # Compute melt weight from HRRR (not a SAR channel anymore,
                # but still needed for temporal aggregation weighting)
                mw_i = hrrr_cache.get(times[i])
                mw_j = hrrr_cache.get(times[j])
                melt_w = 1.0
                if mw_i is not None and mw_j is not None:
                    melt_w = float(np.minimum(mw_i, mw_j).mean())
                elif mw_i is not None:
                    melt_w = float(mw_i.mean())
                elif mw_j is not None:
                    melt_w = float(mw_j.mean())

                results.append({
                    'sar': sar,
                    'track': track_id,
                    't_start': times[i],
                    't_end': times[j],
                    'span_days': span,
                    'melt_weight_mean': melt_w,
                })

    log.info('get_all_season_pairs: %d unique pairs (max span %dd)',
             len(results), max_span_days)
    return results


def get_pair_metadata_and_tracks(
    ds: xr.Dataset,
    max_span_days: int = 60,
    tau_proximity: float = 12.0,
    hrrr_ds: xr.Dataset | None = None,
) -> tuple[list[dict], dict, dict]:
    """Return pair metadata + track data without materializing SAR arrays.

    Use for memory-efficient extraction: compute one pair's SAR at a time
    via extract_single_pair(tracks[meta['track']], meta['i'], meta['j'], ...).

    Returns
    -------
    pair_metas : list of dicts with track, i, j, t_start, t_end, span_days, melt_weight_mean
    tracks : dict from _get_track_data()
    hrrr_cache : dict of melt weights per timestamp
    """
    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        for t in pd.DatetimeIndex(ds.time.values):
            hrrr_cache[t] = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)

    tracks = _get_track_data(ds)
    metas = []
    seen = set()

    for track_id, td in tracks.items():
        times = td['times']
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                span = (times[j] - times[i]).days
                if span > max_span_days:
                    continue

                key = (track_id, str(times[i]), str(times[j]))
                if key in seen:
                    continue
                seen.add(key)

                # Quick coverage check: does VV have data at both timestamps?
                if np.isfinite(td['vv'][j]).sum() < 100:
                    continue

                mw_i = hrrr_cache.get(times[i])
                mw_j = hrrr_cache.get(times[j])
                melt_w = 1.0
                if mw_i is not None and mw_j is not None:
                    melt_w = float(np.minimum(mw_i, mw_j).mean())
                elif mw_i is not None:
                    melt_w = float(mw_i.mean())
                elif mw_j is not None:
                    melt_w = float(mw_j.mean())

                metas.append({
                    'track': track_id,
                    'i': i,
                    'j': j,
                    't_start': times[i],
                    't_end': times[j],
                    'span_days': span,
                    'melt_weight_mean': melt_w,
                })

    log.info('get_pair_metadata_and_tracks: %d pairs, %d tracks (max span %dd)',
             len(metas), len(tracks), max_span_days)
    return metas, tracks, hrrr_cache


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

    # Compute TPI from DEM
    derived = {}
    if 'dem' in ds.data_vars:
        try:
            from sarvalanche.utils.terrain import compute_tpi
            # compute_tpi expects xr.DataArray with projected CRS
            dem_da = ds['dem']
            if ds.rio.crs and ds.rio.crs.is_geographic:
                # Reproject to UTM for terrain computation
                dem_da = dem_da.rio.reproject('EPSG:32606')
                tpi_da = compute_tpi(dem_da, radius_m=300.0)
                # Reproject back
                tpi_da = tpi_da.rio.reproject_match(ds['dem'])
                derived['tpi'] = np.nan_to_num(tpi_da.values.astype(np.float32), nan=0.0)
            else:
                tpi_da = compute_tpi(dem_da, radius_m=300.0)
                derived['tpi'] = np.nan_to_num(tpi_da.values.astype(np.float32), nan=0.0)
            log.info('  Computed TPI from DEM')
        except Exception as e:
            log.warning('  Could not compute TPI: %s', e)
            derived['tpi'] = np.zeros((H, W), dtype=np.float32)

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
