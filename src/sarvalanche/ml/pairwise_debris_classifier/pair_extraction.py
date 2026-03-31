"""Shared pair extraction utilities for training and inference.

Extracts per-track VV/VH pair diffs with valid masks and ANF from a
pre-denoised xr.Dataset. Used by both inference.py and dataset.py.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.features.backscatter_change import backscatter_changes_all_pairs
from sarvalanche.ml.pairwise_debris_classifier.channels import normalize_anf
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear

log = logging.getLogger(__name__)


def _ensure_db(da):
    """Convert DataArray to dB if not already."""
    if check_db_linear(da) != 'dB':
        return linear_to_dB(da)
    return da


def get_track_data(ds: xr.Dataset):
    """Build per-track VV, VH, and ANF lookups from a dataset.

    Returns
    -------
    vv_by_track : dict {track_str: DataArray in dB}
    vh_by_track : dict {track_str: DataArray in dB}
    anf_per_track : dict {track_str: (H, W) normalized ANF}
    anf_raw_per_track : dict {track_str: (H, W) raw ANF}
    """
    H, W = ds.sizes['y'], ds.sizes['x']
    has_anf = 'anf' in ds.data_vars

    vv_by_track = {}
    for track, pol, da in iter_track_pol_combinations(ds, polarizations=['VV']):
        vv_by_track[str(track)] = _ensure_db(da)

    vh_by_track = {}
    for track, pol, da in iter_track_pol_combinations(ds, polarizations=['VH']):
        vh_by_track[str(track)] = _ensure_db(da)

    anf_per_track = {}
    anf_raw_per_track = {}
    for track_str in vv_by_track:
        track_int = int(track_str)
        if has_anf and track_int in ds['anf'].static_track.values:
            anf_arr = np.nan_to_num(
                ds['anf'].sel(static_track=track_int).values.astype(np.float32),
                nan=1.0)
            anf_per_track[track_str] = normalize_anf(anf_arr)
            anf_raw_per_track[track_str] = anf_arr
        else:
            log.warning("Track %s: no ANF data — using uniform ANF=1.0, "
                        "layover masking disabled for this track", track_str)
            anf_per_track[track_str] = np.ones((H, W), dtype=np.float32)

    return vv_by_track, vh_by_track, anf_per_track, anf_raw_per_track


def align_vh_to_vv(vv_diffs, vh_diffs):
    """Align VH pair diffs to VV by matching t_start/t_end timestamps.

    Returns a dict mapping VV pair index → VH pair index, or None if
    VH is unavailable.
    """
    if vh_diffs is None:
        return None

    vh_lookup = {}
    for p in range(vh_diffs.sizes['pair']):
        key = (pd.Timestamp(vh_diffs.t_start.values[p]),
               pd.Timestamp(vh_diffs.t_end.values[p]))
        vh_lookup[key] = p

    alignment = {}
    for p in range(vv_diffs.sizes['pair']):
        key = (pd.Timestamp(vv_diffs.t_start.values[p]),
               pd.Timestamp(vv_diffs.t_end.values[p]))
        if key in vh_lookup:
            alignment[p] = vh_lookup[key]
        else:
            log.warning("VV pair %s→%s has no matching VH pair",
                        key[0].date(), key[1].date())
    return alignment


def extract_pair_diff(vv_diffs, vh_diffs, p, vh_alignment):
    """Extract one pair's VV/VH diffs with valid mask.

    Parameters
    ----------
    vv_diffs : xr.DataArray with pair dim
    vh_diffs : xr.DataArray with pair dim, or None
    p : int — VV pair index
    vh_alignment : dict mapping VV index → VH index, or None

    Returns
    -------
    vv_arr : (H, W) float32, NaN filled to 0
    vh_arr : (H, W) float32, NaN filled to 0
    valid_mask : (H, W) bool — True where both VV and VH have data
    """
    vv_raw = vv_diffs.isel(pair=p).values.astype(np.float32)
    vv_valid = np.isfinite(vv_raw)
    vv_arr = np.nan_to_num(vv_raw, nan=0.0)

    valid_mask = vv_valid
    if vh_alignment is not None and p in vh_alignment:
        vh_raw = vh_diffs.isel(pair=vh_alignment[p]).values.astype(np.float32)
        vh_valid = np.isfinite(vh_raw)
        valid_mask = vv_valid & vh_valid
        vh_arr = np.nan_to_num(vh_raw, nan=0.0)
    else:
        vh_arr = np.zeros_like(vv_arr)

    return vv_arr, vh_arr, valid_mask


def extract_all_pairs(ds: xr.Dataset, max_span_days: int = 60):
    """Extract all pair diffs, metadata, and ANF from a dataset.

    Convenience function that combines get_track_data, backscatter_changes_all_pairs,
    align_vh_to_vv, and extract_pair_diff into one call.

    Returns
    -------
    pair_diffs : list of (vv_diff, vh_diff, valid_mask) tuples
    pair_metas : list of dicts with 'track', 'span_days', 't_start', 't_end'
    anf_per_track : dict {track_str: (H, W) normalized ANF}
    anf_raw_per_track : dict {track_str: (H, W) raw ANF}
    """
    vv_by_track, vh_by_track, anf_per_track, anf_raw_per_track = get_track_data(ds)

    pair_diffs = []
    pair_metas = []

    for track_str, da_vv in vv_by_track.items():
        da_vh = vh_by_track.get(track_str)

        try:
            vv_diffs = backscatter_changes_all_pairs(da_vv, max_span_days=max_span_days)
        except ValueError:
            log.warning("Track %s: no pairs with span <= %dd", track_str, max_span_days)
            continue

        vh_diffs = None
        if da_vh is not None:
            try:
                vh_diffs = backscatter_changes_all_pairs(da_vh, max_span_days=max_span_days)
            except ValueError:
                pass

        vh_alignment = align_vh_to_vv(vv_diffs, vh_diffs)

        n_pairs = vv_diffs.sizes['pair']
        for p in range(n_pairs):
            t_start = pd.Timestamp(vv_diffs.t_start.values[p])
            t_end = pd.Timestamp(vv_diffs.t_end.values[p])
            span = (t_end - t_start).days

            vv_arr, vh_arr, valid_mask = extract_pair_diff(
                vv_diffs, vh_diffs, p, vh_alignment)

            pair_diffs.append((vv_arr, vh_arr, valid_mask))
            pair_metas.append({
                'track': track_str,
                't_start': t_start,
                't_end': t_end,
                'span_days': span,
            })

        log.info("  Track %s: %d pairs", track_str, n_pairs)

    log.info("Total pairs: %d across %d tracks", len(pair_diffs), len(anf_per_track))
    return pair_diffs, pair_metas, anf_per_track, anf_raw_per_track
