"""V4 inference utilities — shared between training and evaluation.

All patch-building functions exactly replicate the training data pipeline
in dataset_inmemory.py. Any change to training must be mirrored here.

Source of truth: V3InMemoryDataset._make_sar_static() and __getitem__()
"""

import numpy as np
from skimage.transform import resize

from sarvalanche.ml.v3.channels import N_SAR, N_STATIC
from sarvalanche.ml.v3.patch_extraction import V3_PATCH_SIZE, normalize_dem_patch


def build_sar_channels(vv_diff, vh_diff, anf_norm, span_days,
                       vv_smooth=None, vh_smooth=None):
    """Build 7-channel SAR array from raw dB diffs.

    Exactly matches dataset_inmemory._make_sar_static() SAR construction.

    Parameters
    ----------
    vv_diff : (H, W) float32 — raw VV dB difference (after - before)
    vh_diff : (H, W) float32 — raw VH dB difference
    anf_norm : (H, W) float32 — ALREADY normalized ANF (via _normalize_anf)
    span_days : int — pair span in days
    vv_smooth : (H, W) float32, optional — TV-denoised VV diff
    vh_smooth : (H, W) float32, optional — TV-denoised VH diff

    Returns
    -------
    (7, H, W) float32 — [change_vv, change_vh, change_cr, anf, proximity,
                          change_vv_smooth, change_vh_smooth]
    """
    vv = vv_diff.astype(np.float32)
    vh = vh_diff.astype(np.float32)
    change_vv = np.sign(vv) * np.log1p(np.abs(vv))
    change_vh = np.sign(vh) * np.log1p(np.abs(vh))
    cr = vh - vv
    change_cr = np.sign(cr) * np.log1p(np.abs(cr))
    prox = np.full(vv.shape, 1.0 / (1.0 + span_days / 12.0), dtype=np.float32)

    if vv_smooth is None:
        from skimage.restoration import denoise_tv_chambolle
        vv_smooth = denoise_tv_chambolle(vv, weight=1.0).astype(np.float32)
        vh_smooth = denoise_tv_chambolle(vh, weight=1.0).astype(np.float32)

    change_vv_s = np.sign(vv_smooth) * np.log1p(np.abs(vv_smooth))
    change_vh_s = np.sign(vh_smooth) * np.log1p(np.abs(vh_smooth))

    # Coverage mask: 1 where real SAR data, 0 where NaN-filled
    coverage = (np.abs(vv) > 1e-6).astype(np.float32)

    return np.stack([change_vv, change_vh, change_cr, anf_norm, prox,
                     change_vv_s, change_vh_s, coverage], axis=0)


def build_patch(sar_scene, static_scene, y0, x0, size):
    """Build a single (SAR + static) patch with per-patch DEM normalization.

    Exactly matches dataset_inmemory._make_sar_static(): extracts the window,
    normalizes DEM per-patch, zero-pads at scene edges.

    Parameters
    ----------
    sar_scene : (N_SAR, H, W) — full scene SAR channels
    static_scene : (N_STATIC, H, W) — full scene static channels (raw, NOT pre-normalized)
    y0, x0 : int — top-left corner (can be negative for edge padding)
    size : int — patch size in pixels

    Returns
    -------
    (N_SAR + N_STATIC, size, size) float32
    """
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    y0c, x0c = max(y0, 0), max(x0, 0)
    y1, x1 = min(y0 + size, H), min(x0 + size, W)

    sar = sar_scene[:, y0c:y1, x0c:x1]
    static = normalize_dem_patch(static_scene[:, y0c:y1, x0c:x1].copy())

    ah, aw = y1 - y0c, x1 - x0c
    if ah < size or aw < size:
        sar_p = np.zeros((N_SAR, size, size), dtype=np.float32)
        sta_p = np.zeros((N_STATIC, size, size), dtype=np.float32)
        py, px = y0c - y0, x0c - x0
        sar_p[:, py:py + ah, px:px + aw] = sar
        sta_p[:, py:py + ah, px:px + aw] = static
        return np.concatenate([sar_p, sta_p], axis=0)

    return np.concatenate([sar, static], axis=0)


def build_v4_inputs(sar_scene, static_scene, y0, x0, patch_size=V3_PATCH_SIZE):
    """Build all 3 v4 inputs (fine, local_ctx, regional NOT included).

    Exactly matches dataset_inmemory.__getitem__() for fine and local_ctx.
    Regional must be precomputed separately via build_regional().

    Returns
    -------
    fine : (C, ps, ps) — full resolution patch
    local_ctx : (C, ps, ps) — 512×512 centered, downsampled to 128×128
    """
    ps = patch_size
    C = N_SAR + N_STATIC

    fine = build_patch(sar_scene, static_scene, y0, x0, ps)

    # Local context: 512×512 centered on the fine patch, downsampled 4×
    ctx_size = ps * 4
    ctx_y0 = y0 - (ctx_size - ps) // 2
    ctx_x0 = x0 - (ctx_size - ps) // 2
    ctx_full = build_patch(sar_scene, static_scene, ctx_y0, ctx_x0, ctx_size)
    local_ctx = ctx_full.reshape(C, ps, 4, ps, 4).mean(axis=(2, 4))

    return fine, local_ctx


def build_regional(sar_scene, static_scene, anf_norm, vv_diff, vh_diff, span_days,
                   target=V3_PATCH_SIZE, vv_smooth=None, vh_smooth=None):
    """Build regional input: whole scene downsampled to (C, target, target).

    Exactly matches build_inmemory_dataset() regional cache construction.
    Note: static is NOT DEM-normalized for regional (training doesn't normalize it).
    SAR channels are recomputed from raw diffs at downsampled resolution.

    Parameters
    ----------
    sar_scene : not used (kept for API consistency)
    static_scene : (N_STATIC, H, W) — raw static channels
    anf_norm : (H, W) — normalized ANF for this track
    vv_diff, vh_diff : (H, W) — raw dB diffs
    span_days : int
    target : int — output size (default 128)
    vv_smooth, vh_smooth : (H, W) optional — pre-computed TV-denoised diffs
    """
    from skimage.restoration import denoise_tv_chambolle

    # Downsample raw diffs then compute SAR channels (matches training order)
    vv_s = resize(vv_diff, (target, target), order=1, preserve_range=True).astype(np.float32)
    vh_s = resize(vh_diff, (target, target), order=1, preserve_range=True).astype(np.float32)
    cv = np.sign(vv_s) * np.log1p(np.abs(vv_s))
    ch = np.sign(vh_s) * np.log1p(np.abs(vh_s))
    cr = vh_s - vv_s
    ccr = np.sign(cr) * np.log1p(np.abs(cr))
    anf_s = resize(anf_norm, (target, target), order=1, preserve_range=True).astype(np.float32)
    prox = np.full((target, target), 1.0 / (1.0 + span_days / 12.0), dtype=np.float32)

    # Smoothed channels
    if vv_smooth is None:
        vv_smooth = denoise_tv_chambolle(vv_diff.astype(np.float32), weight=1.0).astype(np.float32)
        vh_smooth = denoise_tv_chambolle(vh_diff.astype(np.float32), weight=1.0).astype(np.float32)
    vv_sm_s = resize(vv_smooth, (target, target), order=1, preserve_range=True).astype(np.float32)
    vh_sm_s = resize(vh_smooth, (target, target), order=1, preserve_range=True).astype(np.float32)
    cvs = np.sign(vv_sm_s) * np.log1p(np.abs(vv_sm_s))
    chs = np.sign(vh_sm_s) * np.log1p(np.abs(vh_sm_s))

    # Coverage at regional scale (fraction of valid pixels)
    cov = (np.abs(vv_diff) > 1e-6).astype(np.float32)
    cov_r = resize(cov, (target, target), order=1, preserve_range=True).astype(np.float32)

    sar_r = np.stack([cv, ch, ccr, anf_s, prox, cvs, chs, cov_r], axis=0)

    # Static: raw (no DEM normalization), just downsample
    static_r = resize(static_scene.transpose(1, 2, 0), (target, target),
                      order=1, preserve_range=True).transpose(2, 0, 1).astype(np.float32)

    return np.concatenate([sar_r, static_r], axis=0)
