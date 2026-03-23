"""Channel definitions for v3 single-pair debris detector.

v3 simplification: one SAR pair at a time, no set encoder / attention.
The CNN evaluates each pair independently against static terrain context.
Temporal aggregation happens post-inference via the temporal onset detector.

Trimmed based on ablation study (2026-03-19):
  - Removed: vv_magnitude, vh_magnitude, melt_weight (SAR)
  - Removed: fcf, water_mask, release_zones, runout_angle, curvature (static)
  - Added: VH change, cross-ratio change (2026-03-21)
  - Removed d_cr after further ablation showed no benefit
"""

import numpy as np


# ── SAR channels (per pair) ──────────────────────────────────────────

SAR_CHANNELS: list[str] = [
    'change_vv',        # log1p VV dB diff (after - before)
    'change_vh',        # log1p VH dB diff (after - before)
    'change_cr',        # cross-ratio change (VH-VV)_after - (VH-VV)_before
    'anf',              # normalized local incidence angle
    'proximity',        # temporal proximity weight
]

N_SAR: int = len(SAR_CHANNELS)


# ── Static channels ──────────────────────────────────────────────────

STATIC_CHANNELS: list[str] = [
    'slope',
    'aspect_northing',
    'aspect_easting',
    'dem',                          # per-patch min-max normalized
    'cell_counts',
    'tpi',
]

N_STATIC: int = len(STATIC_CHANNELS)

# Total input channels to the CNN
N_INPUT: int = N_SAR + N_STATIC


# ── Normalization ────────────────────────────────────────────────────

STATIC_NORM: dict[str, dict] = {
    'slope': {'scale': 0.6},
    'cell_counts': {'log1p': True, 'scale': 5.0},
    'tpi': {'scale': 50.0},
}


def normalize_static_channel(arr: np.ndarray, var: str) -> np.ndarray:
    """Apply normalization to a static channel array."""
    cfg = STATIC_NORM.get(var)
    if not cfg:
        return arr
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return arr
