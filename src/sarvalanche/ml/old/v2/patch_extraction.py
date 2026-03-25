"""Scene-level patch extraction for v2 format.

Builds per-track/pol SAR backscatter change maps and static terrain channels
from a dataset, suitable for the v2 CNN architecture.

v2.1 adds per-pair mode: instead of feeding pre-pooled empirical change maps,
individual crossing pairs are fed as separate set elements so the CNN learns
which temporal pairs are informative.
"""

import logging
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, sobel

from sarvalanche.ml.v2.channels import (
    N_STATIC,
    STATIC_CHANNELS,
    normalize_static_channel,
)


log = logging.getLogger(__name__)

V2_PATCH_SIZE = 128

# Background smoothing sigma in pixels (~2 km at 30 m resolution)
_BG_SIGMA_PX = 67


def _normalized_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with normalized convolution to handle no-data (zero) regions.

    Uses the trick: smoothed = blur(arr * mask) / blur(mask), so only valid
    pixels contribute to the local average. Edges and masked regions are
    handled correctly.
    """
    valid = (arr != 0).astype(np.float32)
    blurred_vals = gaussian_filter(arr * valid, sigma=sigma, mode='constant')
    blurred_mask = gaussian_filter(valid, sigma=sigma, mode='constant')
    # Avoid division by zero where no valid pixels contribute
    blurred_mask = np.maximum(blurred_mask, 1e-6)
    return blurred_vals / blurred_mask


def _edge_magnitude(arr: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude of a 2D array, normalized to ~[0, 1]."""
    sx = sobel(arr, axis=0)
    sy = sobel(arr, axis=1)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    # Normalize by robust percentile to get ~[0, 1] range
    p99 = np.percentile(mag[mag > 0], 99) if (mag > 0).any() else 1.0
    return np.clip(mag / max(p99, 1e-6), 0.0, 1.0)


def normalize_anf(arr: np.ndarray) -> np.ndarray:
    """Normalize ANF to [0, 1] where 1 = best quality (low ANF).

    ANF ranges from ~1 (good) to 100s (poor). Log-compress and invert
    so that good viewing geometry → high values.
    """
    return 1.0 / (1.0 + np.log1p(arr))


def normalize_magnitude(arr: np.ndarray) -> np.ndarray:
    """Normalize dB backscatter magnitude to ~[0, 1].

    Sentinel-1 RTC backscatter typically ranges from -25 to +10 dB.
    Maps linearly to [0, 1] then clips.
    """
    return np.clip((arr + 25.0) / 35.0, 0.0, 1.0)


def get_per_track_pol_changes(
    ds: xr.Dataset,
    n_channels: int = 4,
) -> list[tuple[str, str, np.ndarray]]:
    """Get per-track/pol empirical backscatter change maps from dataset.

    Looks for variables named d_{track}_{pol}_empirical that are produced
    by calculate_empirical_backscatter_probability.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with per-track/pol empirical change maps and anf.
    n_channels : int
        Number of channels per SAR map. 2 = [change, ANF], 4 = [change, ANF,
        anomaly, edges]. Default 4.

    Returns:
    -------
    list of (track, pol, change_map) tuples where change_map is
    (n_channels, H, W) float32.
    """
    pattern = re.compile(r'^d_(\d+)_(V[VH])_empirical$')
    has_anf = 'anf' in ds.data_vars
    results = []
    for var in sorted(ds.data_vars):
        m = pattern.match(var)
        if m:
            track, pol = m.group(1), m.group(2)
            arr = np.nan_to_num(ds[var].values.astype(np.float32), nan=0.0)
            # Compress dynamic range so winter (~3 dB) and spring (~7 dB)
            # changes are handled consistently by the set encoder
            arr = np.sign(arr) * np.log1p(np.abs(arr))

            # Get matching ANF for this track
            if has_anf:
                track_int = int(track)
                anf_arr = np.nan_to_num(
                    ds['anf'].sel(static_track=track_int).values.astype(np.float32),
                    nan=1.0,
                )
                anf_norm = normalize_anf(anf_arr)
            else:
                anf_norm = np.ones_like(arr)

            if n_channels == 2:
                stacked = np.stack([arr, anf_norm], axis=0)  # (2, H, W)
            else:
                # High-pass anomaly: change minus smoothed background
                background = _normalized_gaussian_blur(arr, sigma=_BG_SIGMA_PX)
                anomaly = arr - background
                # Edge magnitude of the change map
                edges = _edge_magnitude(arr)
                stacked = np.stack([arr, anf_norm, anomaly, edges], axis=0)  # (4, H, W)

            results.append((track, pol, stacked))
    return results


# ---------------------------------------------------------------------------
# v2.1: Per-pair change maps (no temporal pooling)
# ---------------------------------------------------------------------------

def _hrrr_pdd_melt_weight(hrrr_ds, sar_date, hrrr_times, pdd_threshold=0.1):
    """Compute per-pixel melt weight from HRRR PDD for a SAR date.

    Returns (H, W) float32: 0 = warm/melting, 1 = cold/trusted.
    Returns None if no HRRR data available for this date.
    """
    from scipy.ndimage import gaussian_filter as _gf

    time_diffs = np.abs(hrrr_times - sar_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 2:
        return None

    if 'pdd_24h' in hrrr_ds and 't2m_max' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        t2m = hrrr_ds['t2m_max'].isel(time=ci).values

        pdd_smooth = _gf(pdd, sigma=15, mode='nearest')
        t2m_smooth = _gf(t2m, sigma=15, mode='nearest')

        # PDD weight: any positive degree-day energy suppresses
        pdd_weight = np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0)

        # Temperature weight: suppress near-freezing even if PDD=0
        # Captures solar melt and recent refreeze that HRRR misses
        # T <= -8°C → 1.0 (fully trusted, well below freezing)
        # T  = -5°C → 0.5 (partial suppression)
        # T >= -3°C → 0.0 (recent melt/refreeze likely)
        t2m_weight = np.clip((-t2m_smooth - 3.0) / 5.0, 0.0, 1.0)

        # Take the minimum — either criterion suppresses
        melt_weight = np.minimum(pdd_weight, t2m_weight)
        return melt_weight.astype(np.float32)
    elif 'pdd_24h' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        pdd_smooth = _gf(pdd, sigma=15, mode='nearest')
        return np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0).astype(np.float32)
    elif 't2m_max' in hrrr_ds:
        t2m = hrrr_ds['t2m_max'].isel(time=ci).values
        t2m_smooth = _gf(t2m, sigma=15, mode='nearest')
        return np.clip((-t2m_smooth - 1.0) / 4.0, 0.0, 1.0).astype(np.float32)
    return None


def get_per_pair_changes(
    ds: xr.Dataset,
    reference_date,
    max_pairs: int = 4,
    tau_proximity: float = 12.0,
    hrrr_ds: xr.Dataset | None = None,
) -> list[np.ndarray]:
    """Get individual crossing-pair change maps for all track/pol combos.

    Instead of a single temporally-weighted average per track/pol, returns
    each crossing pair as its own SAR array:
        ch 0: log1p-compressed dB change (after - before)
        ch 1: normalized ANF for the track
        ch 2: temporal proximity = 1/(1 + span_days/tau)
        ch 3: melt_weight (0=warm, 1=cold) — only if hrrr_ds provided

    Parameters
    ----------
    ds : xr.Dataset
        Season dataset with VV, VH (time, y, x) in dB, track coord, anf.
    reference_date : str or Timestamp
        Avalanche reference date.
    max_pairs : int
        Max crossing pairs to keep per track/pol (sorted by tightest span).
    tau_proximity : float
        Scale for temporal proximity channel (days).
    hrrr_ds : xr.Dataset, optional
        HRRR temperature dataset with pdd_24h or t2m_max variables.
        If provided, a 4th melt_weight channel is added per pair.

    Returns
    -------
    list of (C, H, W) float32 arrays (C=3 or 4), one per selected pair.
    """
    from sarvalanche.utils.generators import iter_track_pol_combinations

    ref = pd.Timestamp(reference_date)
    has_anf = 'anf' in ds.data_vars
    n_ch = 4 if hrrr_ds is not None else 3

    # Precompute HRRR melt weights for all SAR times
    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        times_all = pd.DatetimeIndex(ds.time.values)
        for t in times_all:
            mw = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)
            hrrr_cache[t] = mw  # None if no data

    results = []

    for track, pol, da in iter_track_pol_combinations(ds):
        # Get all crossing pairs
        times = pd.DatetimeIndex(da.time.values)
        pairs = []
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                ti, tj = times[i], times[j]
                if ti <= ref < tj:
                    span_days = (tj - ti).days
                    pairs.append((i, j, span_days))

        if not pairs:
            continue

        # Sort by tightest span, take top K
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:max_pairs]

        # Get ANF for this track
        if has_anf:
            track_int = int(track)
            if track_int in ds['anf'].static_track.values:
                anf_arr = np.nan_to_num(
                    ds['anf'].sel(static_track=track_int).values.astype(np.float32),
                    nan=1.0,
                )
                anf_norm = normalize_anf(anf_arr)
            else:
                anf_norm = np.ones(
                    (ds.sizes['y'], ds.sizes['x']), dtype=np.float32,
                )
        else:
            anf_norm = np.ones(
                (ds.sizes['y'], ds.sizes['x']), dtype=np.float32,
            )

        for i, j, span_days in pairs:
            # Raw dB difference
            diff = (da.isel(time=j).values - da.isel(time=i).values).astype(np.float32)
            diff = np.nan_to_num(diff, nan=0.0)
            # log1p compression
            diff = np.sign(diff) * np.log1p(np.abs(diff))

            # Temporal proximity: tighter pair = higher value
            proximity = np.full_like(diff, 1.0 / (1.0 + span_days / tau_proximity))

            channels = [diff, anf_norm, proximity]

            # Melt weight: min of both endpoints
            if hrrr_ds is not None:
                ti, tj = times[i], times[j]
                mw_i = hrrr_cache.get(ti)
                mw_j = hrrr_cache.get(tj)
                melt_w = np.ones_like(diff)
                if mw_i is not None:
                    melt_w = np.minimum(melt_w, mw_i)
                if mw_j is not None:
                    melt_w = np.minimum(melt_w, mw_j)
                channels.append(melt_w)

            stacked = np.stack(channels, axis=0)  # (3 or 4, H, W)
            results.append(stacked)

    log.info('get_per_pair_changes: %d pairs (%d ch) for date %s',
             len(results), n_ch, ref.date())
    return results


def precompute_pair_scene_arrays(
    ds: xr.Dataset,
    reference_date,
    max_pairs: int = 4,
    hrrr_ds: xr.Dataset | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute full-scene per-pair SAR arrays + static stack.

    Parameters
    ----------
    hrrr_ds : xr.Dataset, optional
        HRRR temperature data. If provided, adds melt_weight as 4th SAR
        channel and uses melt-filtered d_empirical in static stack.

    Returns
    -------
    sar_scene : (N_pairs, C, H, W) float32 (C=3 or 4)
    static_scene : (N_STATIC, H, W) float32
    """
    pair_maps = get_per_pair_changes(ds, reference_date, max_pairs=max_pairs,
                                     hrrr_ds=hrrr_ds)
    n_ch = 4 if hrrr_ds is not None else 3
    if not pair_maps:
        H, W = ds.sizes['y'], ds.sizes['x']
        sar_scene = np.zeros((1, n_ch, H, W), dtype=np.float32)
    else:
        sar_scene = np.stack(pair_maps, axis=0)
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    return sar_scene, static_scene


def precompute_combo_scene_arrays(
    ds: xr.Dataset,
    reference_date,
    max_pairs: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute combined pooled + per-pair SAR arrays + static stack.

    Feeds both temporally-pooled change maps (2-ch padded to 3-ch with
    proximity=1.0) and individual crossing-pair maps (3-ch) into a single
    set. The set encoder + attention learns which elements are informative.

    Returns
    -------
    sar_scene : (N_pooled + N_pairs, 3, H, W) float32
    static_scene : (N_STATIC, H, W) float32
    """
    H, W = ds.sizes['y'], ds.sizes['x']

    # Pooled maps: 2-ch → pad to 3-ch with proximity=1.0
    pooled_results = get_per_track_pol_changes(ds, n_channels=2)
    pooled_maps = []
    for track, pol, arr_2ch in pooled_results:
        # arr_2ch is (2, H, W) → pad with ones to (3, H, W)
        proximity = np.ones((1, H, W), dtype=np.float32)
        arr_3ch = np.concatenate([arr_2ch, proximity], axis=0)
        pooled_maps.append(arr_3ch)

    # Per-pair maps: already (3, H, W)
    pair_maps = get_per_pair_changes(ds, reference_date, max_pairs=max_pairs)

    all_maps = pooled_maps + pair_maps
    if not all_maps:
        sar_scene = np.zeros((1, 3, H, W), dtype=np.float32)
    else:
        sar_scene = np.stack(all_maps, axis=0)

    log.info(
        'precompute_combo_scene_arrays: %d pooled + %d pairs = %d total SAR maps',
        len(pooled_maps), len(pair_maps), len(all_maps),
    )

    static_scene = build_static_stack(ds)
    return sar_scene, static_scene


DEM_CHANNEL: int = STATIC_CHANNELS.index('dem')

# Channels that need per-patch normalization instead of global normalization
_PATCH_NORM_CHANNELS: set[str] = {'dem'}


def _compute_d_cr(ds: xr.Dataset, H: int, W: int) -> np.ndarray | None:
    """Compute cross-ratio change: d_VH - d_VV per track, weighted by w_resolution.

    Cross-ratio (VH - VV in dB) changes differently for wet snow vs debris:
    wet snow decreases CR (VH drops more), debris increases or keeps CR stable.

    Per-track contributions are weighted by spatial resolution weights so that
    tracks with better viewing geometry contribute more.
    """
    pattern_vh = re.compile(r'^d_(\d+)_VH_empirical$')
    tracks = []
    for var in ds.data_vars:
        m = pattern_vh.match(var)
        if m:
            track = m.group(1)
            vv_var = f'd_{track}_VV_empirical'
            if vv_var in ds.data_vars:
                tracks.append(track)

    if not tracks:
        return None

    has_w_res = 'w_resolution' in ds.data_vars and 'static_track' in ds['w_resolution'].dims
    weighted_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for track in tracks:
        d_vh = np.nan_to_num(ds[f'd_{track}_VH_empirical'].values.astype(np.float32), nan=0.0)
        d_vv = np.nan_to_num(ds[f'd_{track}_VV_empirical'].values.astype(np.float32), nan=0.0)
        d_cr = d_vh - d_vv

        if has_w_res:
            track_int = int(track)
            if track_int in ds['w_resolution'].static_track.values:
                w = np.nan_to_num(
                    ds['w_resolution'].sel(static_track=track_int).values.astype(np.float32),
                    nan=0.0,
                )
            else:
                w = np.ones((H, W), dtype=np.float32)
        else:
            w = np.ones((H, W), dtype=np.float32)

        weighted_sum += d_cr * w
        weight_sum += w

    weight_sum = np.maximum(weight_sum, 1e-6)
    return (weighted_sum / weight_sum).astype(np.float32)


def _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W):
    """Compute melt-filtered d_empirical: per-pair diffs with warm pairs downweighted.

    Replicates the full pipeline (per track/pol crossing pairs, temporal weights,
    resolution+pol combination) but multiplies each pair's weight by melt_weight
    at both endpoints.

    Returns (H, W) float32 array, or None if computation fails.
    """
    from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
    from sarvalanche.preprocessing.radiometric import linear_to_dB
    from sarvalanche.utils.generators import iter_track_pol_combinations
    from sarvalanche.utils.validation import check_db_linear
    from sarvalanche.weights.polarizations import get_polarization_weights
    from sarvalanche.weights.temporal import get_temporal_weights

    if 'd_empirical' not in ds or 'w_temporal' not in ds:
        return None

    # Find the reference date from w_temporal (peak weight)
    w_t = ds['w_temporal'].values
    ref_idx = np.argmax(w_t)
    avalanche_date = ds.time.values[ref_idx]
    tau = 6  # match standard pipeline

    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    # Precompute melt weights for all SAR times
    mw_cache = {}
    for t in pd.DatetimeIndex(ds.time.values):
        mw_cache[t] = _hrrr_pdd_melt_weight(hrrr_ds, t, hrrr_times)

    filtered_results = []
    res_weights = []
    pol_weights = []

    for track, pol, da in iter_track_pol_combinations(ds):
        if check_db_linear(da) != 'dB':
            da = linear_to_dB(da)
        try:
            diffs = backscatter_changes_crossing_date(da, avalanche_date)
        except ValueError:
            continue

        w_pair = get_temporal_weights(diffs['t_start'], diffs['t_end'], tau_days=tau)
        diffs_np = diffs.values
        n_pairs = len(diffs['pair'])
        spatial = diffs_np[0].shape

        # Build per-pixel weights with melt downweighting
        w_3d = np.broadcast_to(w_pair.values[:, None, None], (n_pairs,) + spatial).copy()
        # Zero NaN diffs
        nan_mask = np.isnan(diffs_np)
        w_3d[nan_mask] = 0.0
        # Apply melt weights
        for p in range(n_pairs):
            ts = pd.Timestamp(diffs['t_start'].values[p])
            te = pd.Timestamp(diffs['t_end'].values[p])
            pair_mw = np.ones(spatial, dtype=np.float64)
            if mw_cache.get(ts) is not None:
                pair_mw = np.minimum(pair_mw, mw_cache[ts])
            if mw_cache.get(te) is not None:
                pair_mw = np.minimum(pair_mw, mw_cache[te])
            w_3d[p] *= pair_mw

        # Renormalize
        w_sum = w_3d.sum(axis=0, keepdims=True)
        with np.errstate(invalid='ignore'):
            w_norm = np.where(w_sum > 0, w_3d / w_sum, 0.0)
        d_track = np.nansum(diffs_np * w_norm, axis=0)
        # Fallback to standard where no cold pairs
        no_data = w_sum.squeeze() == 0
        if 'd_empirical' in ds:
            d_track[no_data] = np.nan_to_num(ds['d_empirical'].values[no_data], nan=0.0)

        filtered_results.append(d_track)
        # Resolution weight (scalar mean for this track)
        if 'w_resolution' in ds and int(track) in ds['w_resolution'].static_track.values:
            rw = float(ds['w_resolution'].sel(static_track=int(track)).mean())
        else:
            rw = 1.0
        res_weights.append(rw)
        pol_weights.append(get_polarization_weights(pol))

    if not filtered_results:
        return None

    # Combine across track/pol
    rw = np.array(res_weights)
    pw = np.array(pol_weights)
    cw = rw * pw
    cw = cw / cw.sum()

    result = np.zeros((H, W), dtype=np.float32)
    w_total = np.zeros((H, W), dtype=np.float32)
    for i, d in enumerate(filtered_results):
        valid = np.isfinite(d)
        result[valid] += d[valid].astype(np.float32) * cw[i]
        w_total[valid] += cw[i]
    with np.errstate(invalid='ignore'):
        result = np.where(w_total > 0, result / w_total, 0.0)

    return result.astype(np.float32)


def build_static_stack(ds: xr.Dataset, hrrr_ds: xr.Dataset | None = None) -> np.ndarray:
    """Build (N_STATIC, H, W) normalized static terrain stack.

    All channels except DEM are globally normalized. DEM is left raw here
    and must be per-patch normalized after slicing (see normalize_dem_patch).

    Parameters
    ----------
    ds : xr.Dataset
    hrrr_ds : xr.Dataset, optional
        If provided, d_empirical channel uses melt-filtered version.

    Returns:
    -------
    (N_STATIC, H, W) float32 array.
    """
    H, W = ds.sizes['y'], ds.sizes['x']
    stack = np.zeros((N_STATIC, H, W), dtype=np.float32)

    # Pre-compute aspect decomposition if needed
    aspect_derived = {}
    if 'aspect' in ds.data_vars and (
        'aspect_northing' in STATIC_CHANNELS or 'aspect_easting' in STATIC_CHANNELS
    ):
        aspect = np.nan_to_num(ds['aspect'].values.astype(np.float32), nan=0.0)
        aspect_derived['aspect_northing'] = np.cos(aspect)
        aspect_derived['aspect_easting'] = np.sin(aspect)

    # Pre-compute cross-ratio change: d_cr = d_VH - d_VV per track, then average
    derived = {}
    if 'd_cr' in STATIC_CHANNELS:
        d_cr = _compute_d_cr(ds, H, W)
        if d_cr is not None:
            derived['d_cr'] = d_cr

    # Melt-filtered d_empirical + residual if HRRR available
    if hrrr_ds is not None:
        d_emp_filtered = _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W)
        if d_emp_filtered is not None:
            derived['d_empirical_melt_filtered'] = d_emp_filtered
            # Residual = standard - filtered (isolates the melt signal)
            d_emp_standard = np.nan_to_num(
                ds['d_empirical'].values.astype(np.float32), nan=0.0,
            ) if 'd_empirical' in ds else np.zeros((H, W), dtype=np.float32)
            derived['d_empirical_melt_residual'] = (d_emp_standard - d_emp_filtered)

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
    """Apply per-patch min-max normalization to the DEM channel in-place.

    Maps DEM values within the patch to [0, 1] where 0 = lowest elevation
    and 1 = highest elevation in the patch.

    Parameters
    ----------
    static_patch : (N_STATIC, H, W) float32

    Returns:
    -------
    (N_STATIC, H, W) float32 with DEM channel normalized.
    """
    dem = static_patch[DEM_CHANNEL]
    dem_min = dem.min()
    dem_max = dem.max()
    dem_range = dem_max - dem_min
    if dem_range > 1e-6:
        static_patch[DEM_CHANNEL] = (dem - dem_min) / dem_range
    else:
        static_patch[DEM_CHANNEL] = 0.0
    return static_patch


def precompute_scene_arrays(
    ds: xr.Dataset,
    n_channels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute full-scene SAR and static arrays for fast patch slicing.

    Parameters
    ----------
    ds : xr.Dataset
    n_channels : int
        SAR channels per track/pol (2 or 4).

    Returns
    -------
    sar_scene : (N_tracks, n_channels, H, W) float32
    static_scene : (N_STATIC, H, W) float32
    """
    track_pol_maps = get_per_track_pol_changes(ds, n_channels=n_channels)
    if not track_pol_maps:
        H, W = ds.sizes['y'], ds.sizes['x']
        sar_scene = np.zeros((1, n_channels, H, W), dtype=np.float32)
    else:
        sar_scene = np.stack([arr for _, _, arr in track_pol_maps], axis=0)
    static_scene = build_static_stack(ds)
    return sar_scene, static_scene


def slice_v2_patch(
    sar_scene: np.ndarray,
    static_scene: np.ndarray,
    y0: int,
    x0: int,
    patch_size: int = V2_PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a patch from precomputed scene arrays (fast, no recomputation)."""
    sar = sar_scene[:, :, y0:y0 + patch_size, x0:x0 + patch_size]
    static = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
    static = normalize_dem_patch(static)
    return sar, static


def build_v2_patch(
    ds: xr.Dataset,
    y0: int,
    x0: int,
    patch_size: int = V2_PATCH_SIZE,
    n_channels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a v2-format patch at pixel coordinates (y0, x0).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with per-track/pol d_*_empirical and static variables.
    y0, x0 : int
        Top-left pixel coordinates.
    patch_size : int
        Patch size in pixels (default 128).
    n_channels : int
        SAR channels per track/pol (2 or 4).

    Returns:
    -------
    sar_maps : (N, n_channels, patch_size, patch_size) float32
    static : (N_STATIC, patch_size, patch_size) float32
        Normalized static terrain channels.
    """
    track_pol_maps = get_per_track_pol_changes(ds, n_channels=n_channels)

    if not track_pol_maps:
        log.warning('No per-track/pol change maps found in dataset')
        sar_maps = np.zeros((1, n_channels, patch_size, patch_size), dtype=np.float32)
    else:
        sar_maps = np.stack([
            arr[:, y0:y0 + patch_size, x0:x0 + patch_size]
            for _, _, arr in track_pol_maps
        ], axis=0)

    static_full = build_static_stack(ds)
    static = static_full[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
    static = normalize_dem_patch(static)

    return sar_maps, static


