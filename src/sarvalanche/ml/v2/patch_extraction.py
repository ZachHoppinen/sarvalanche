"""Scene-level patch extraction for v2 format.

Builds per-track/pol SAR backscatter change maps and static terrain channels
from a dataset, suitable for the v2 CNN architecture.
"""

import logging
import re

import numpy as np
import xarray as xr

from sarvalanche.ml.v2.channels import (
    N_STATIC,
    STATIC_CHANNELS,
    normalize_static_channel,
)


log = logging.getLogger(__name__)

V2_PATCH_SIZE = 128


def normalize_anf(arr: np.ndarray) -> np.ndarray:
    """Normalize ANF to [0, 1] where 1 = best quality (low ANF).

    ANF ranges from ~1 (good) to 100s (poor). Log-compress and invert
    so that good viewing geometry → high values.
    """
    return 1.0 / (1.0 + np.log1p(arr))


def get_per_track_pol_changes(ds: xr.Dataset) -> list[tuple[str, str, np.ndarray]]:
    """Get per-track/pol empirical backscatter change maps from dataset.

    Looks for variables named d_{track}_{pol}_empirical that are produced
    by calculate_empirical_backscatter_probability.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with per-track/pol empirical change maps and anf.

    Returns:
    -------
    list of (track, pol, change_map) tuples where change_map is
    (2, H, W) float32 — channel 0 is log1p-compressed backscatter change,
    channel 1 is normalized ANF for the matching track.
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

            stacked = np.stack([arr, anf_norm], axis=0)  # (2, H, W)
            results.append((track, pol, stacked))
    return results


DEM_CHANNEL: int = STATIC_CHANNELS.index('dem')

# Channels that need per-patch normalization instead of global normalization
_PATCH_NORM_CHANNELS: set[str] = {'dem'}


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

    for ch, var in enumerate(STATIC_CHANNELS):
        if var in aspect_derived:
            stack[ch] = aspect_derived[var]
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


def build_v2_patch(
    ds: xr.Dataset,
    y0: int,
    x0: int,
    patch_size: int = V2_PATCH_SIZE,
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

    Returns:
    -------
    sar_maps : (N, 2, patch_size, patch_size) float32
        Per-track/pol backscatter change + ANF maps.
    static : (N_STATIC, patch_size, patch_size) float32
        Normalized static terrain channels.
    """
    track_pol_maps = get_per_track_pol_changes(ds)

    if not track_pol_maps:
        log.warning('No per-track/pol change maps found in dataset')
        sar_maps = np.zeros((1, 2, patch_size, patch_size), dtype=np.float32)
    else:
        sar_maps = np.stack([
            arr[:, y0:y0 + patch_size, x0:x0 + patch_size]
            for _, _, arr in track_pol_maps
        ], axis=0)

    static_full = build_static_stack(ds)
    static = static_full[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
    static = normalize_dem_patch(static)

    return sar_maps, static


