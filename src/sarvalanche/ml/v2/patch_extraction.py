"""Scene-level patch extraction for v2 format.

Builds per-track/pol SAR backscatter change maps and static terrain channels
from a dataset, suitable for the v2 CNN architecture.
"""

import logging
import re

import numpy as np
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


def build_static_stack(ds: xr.Dataset) -> np.ndarray:
    """Build (N_STATIC, H, W) normalized static terrain stack.

    All channels except DEM are globally normalized. DEM is left raw here
    and must be per-patch normalized after slicing (see normalize_dem_patch).

    Parameters
    ----------
    ds : xr.Dataset

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


