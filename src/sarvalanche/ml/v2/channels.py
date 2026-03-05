"""Channel definitions for v2 two-pass CNN architecture."""

import numpy as np


STATIC_CHANNELS: list[str] = [
    'fcf',
    'slope',
    'cell_counts',
    'release_zones',
    'runout_angle',
    'water_mask',
    'd_empirical',
    'aspect_northing',
    'aspect_easting',
    'dem',  # per-patch min-max normalized in patch_extraction
]

N_STATIC: int = len(STATIC_CHANNELS)

STATIC_NORM: dict[str, dict] = {
    'fcf': {'scale': 100.0},
    'slope': {'scale': 0.6},
    'cell_counts': {'log1p': True, 'scale': 5.0},
    'runout_angle': {'scale': np.pi},
    'd_empirical': {'scale': 5.0},
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
