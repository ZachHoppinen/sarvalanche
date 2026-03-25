"""Channel definitions for pairwise debris detector.

Uses pre-denoised SAR imagery (via preprocess_rtc), so the raw dB diff
is already smooth — no separate TV-smoothed channels needed.
"""

import numpy as np


# ── SAR channels (per pair) ──────────────────────────────────────────

SAR_CHANNELS: list[str] = [
    'change_vv',   # sign(x) * log1p(|x|) of VV dB diff. Typical range [-2, 3].
    'change_vh',   # sign(x) * log1p(|x|) of VH dB diff. Typical range [-2, 3].
    'change_cr',   # sign(x) * log1p(|x|) of cross-ratio diff. Typical range [-1.5, 1.5].
    'anf',         # 1/(1+log1p(anf)). Bounded [0, 1]. Per-pixel, per-track.
]

N_SAR: int = len(SAR_CHANNELS)


# ── Static channels ──────────────────────────────────────────────────

STATIC_CHANNELS: list[str] = [
    'slope',
    'aspect_northing',
    'aspect_easting',
    'cell_counts',
    'tpi',
]

N_STATIC: int = len(STATIC_CHANNELS)

# Total input channels to the CNN
N_INPUT: int = N_SAR + N_STATIC


# ── Normalization ────────────────────────────────────────────────────

STATIC_NORM: dict[str, dict] = {
    'slope': {'scale': 0.6},                     # radians, max ~0.6 for steep terrain
    'cell_counts': {'log1p': True, 'scale': 5.0},  # log1p compresses long tail
    'tpi': {'robust_scale': True},                # scale by IQR — TPI range depends on DEM resolution and kernel
}


def sign_log1p(x: np.ndarray) -> np.ndarray:
    """Symmetric log1p transform: sign(x) * log1p(|x|).

    Compresses large values while preserving sign. Used for SAR dB diffs
    where the raw range is roughly [-10, +15] dB but most values are [-3, +5].
    """
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def normalize_anf(arr: np.ndarray) -> np.ndarray:
    """Normalize ANF: 1 = best quality (low ANF), 0 = worst.

    ANF (area normalization factor) varies per-pixel based on local terrain
    slope relative to radar look angle. Higher ANF = more distortion.
    """
    return (1.0 / (1.0 + np.log1p(arr))).astype(np.float32)


def normalize_static_channel(arr: np.ndarray, var: str) -> np.ndarray:
    """Apply normalization to a static channel array.

    Always returns a new array (never mutates input).
    """
    cfg = STATIC_NORM.get(var)
    if not cfg:
        return arr.copy()
    arr = arr.copy()
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    if cfg.get('robust_scale'):
        valid = arr[np.isfinite(arr) & (arr != 0)]
        if len(valid) > 100:
            q25, q75 = np.percentile(valid, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-6:
                arr = arr / iqr
    elif cfg.get('scale'):
        arr = arr / cfg['scale']
    return arr
