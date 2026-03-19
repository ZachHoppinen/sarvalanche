"""Channel definitions for v3 single-pair debris detector.

v3 simplification: one SAR pair at a time, no set encoder / attention.
The CNN evaluates each pair independently against static terrain context.
Temporal aggregation happens post-inference via the temporal onset detector.

SAR channels (per pair):
  - change: log1p-compressed dB difference (after - before)
  - anf: normalized local incidence angle (viewing quality)
  - proximity: temporal proximity = 1/(1 + span_days/tau)
  - melt_weight: HRRR PDD-based melt trust (0=warm, 1=cold)
  - vv_magnitude: mean absolute VV backscatter of pair (dB, normalized)
  - vh_magnitude: mean absolute VH backscatter of pair (dB, normalized)
  - cross_ratio: per-pair VH-VV in dB (wet snow indicator)

Static channels (same for all pairs on a given date):
  - slope
  - aspect_northing (cos)
  - aspect_easting (sin)
  - dem (per-patch min-max normalized)
  - fcf (forest cover fraction)
  - cell_counts (FlowPy runout)
  - release_zones (FlowPy)
  - runout_angle (FlowPy)
  - water_mask
  - curvature
  - tpi (topographic position index)
  - d_empirical_melt_filtered (pooled cold-pair signal, guides the model)
  - d_cr (pooled cross-ratio change)
"""

import numpy as np


# ── SAR channels (per pair) ──────────────────────────────────────────

SAR_CHANNELS: list[str] = [
    'change',           # log1p dB diff (after - before)
    'anf',              # normalized local incidence angle
    'proximity',        # temporal proximity weight
    'melt_weight',      # HRRR PDD melt trust (0=warm, 1=cold)
    'vv_magnitude',     # mean VV of pair (dB, normalized)
    'vh_magnitude',     # mean VH of pair (dB, normalized)
    'cross_ratio',      # per-pair VH-VV dB (wet snow indicator)
]

N_SAR: int = len(SAR_CHANNELS)


# ── Static channels ──────────────────────────────────────────────────

STATIC_CHANNELS: list[str] = [
    'slope',
    'aspect_northing',
    'aspect_easting',
    'dem',                          # per-patch min-max normalized
    'fcf',
    'cell_counts',
    'release_zones',
    'runout_angle',
    'water_mask',
    'curvature',
    'tpi',
    'd_empirical_melt_filtered',    # pooled cold-pair signal
    'd_cr',                         # pooled cross-ratio change
]

N_STATIC: int = len(STATIC_CHANNELS)

# Total input channels to the CNN
N_INPUT: int = N_SAR + N_STATIC


# ── Normalization ────────────────────────────────────────────────────

STATIC_NORM: dict[str, dict] = {
    'fcf': {'scale': 100.0},
    'slope': {'scale': 0.6},
    'cell_counts': {'log1p': True, 'scale': 5.0},
    'runout_angle': {'scale': np.pi},
    'd_empirical_melt_filtered': {'log1p': True, 'scale': 5.0},
    'd_cr': {'log1p': True, 'scale': 3.0},
    'curvature': {'scale': 0.01},       # typical curvature range
    'tpi': {'scale': 50.0},             # typical TPI range in meters
}

SAR_NORM: dict[str, dict] = {
    # change: already log1p compressed in extraction
    # anf: already normalized in extraction
    # proximity: already 0-1
    # melt_weight: already 0-1
    'vv_magnitude': {'offset': 25.0, 'scale': 35.0},   # map [-25,+10] dB → [0,1]
    'vh_magnitude': {'offset': 25.0, 'scale': 35.0},
    'cross_ratio': {'log1p': True, 'scale': 3.0},
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


def normalize_sar_channel(arr: np.ndarray, var: str) -> np.ndarray:
    """Apply normalization to a SAR channel array."""
    cfg = SAR_NORM.get(var)
    if not cfg:
        return arr
    offset = cfg.get('offset', 0)
    if offset:
        arr = arr + offset
    if cfg.get('log1p'):
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    scale = cfg.get('scale')
    if scale:
        arr = arr / scale
    return np.clip(arr, 0, 1) if offset else arr
