"""Channel definitions for Siamese debris classifier.

Unlike the pairwise classifier which feeds pre-computed dB diffs,
the Siamese model receives raw pre/post dB images per branch and
learns what features to diff internally.

Each branch receives: VV (dB), VH (dB), ANF (normalized).
Static channels are shared (not duplicated per branch).
"""

import numpy as np

# ── Per-branch SAR channels (applied to both pre and post) ──────────

BRANCH_CHANNELS: list[str] = [
    'vv',    # Raw dB backscatter
    'vh',    # Raw dB backscatter
    'anf',   # 1/(1+log1p(anf)). Bounded [0, 1]. Per-track.
]

N_BRANCH: int = len(BRANCH_CHANNELS)


# ── Static channels (shared, appended to fused features) ────────────

STATIC_CHANNELS: list[str] = [
    'slope',
    'aspect_northing',
    'aspect_easting',
    'cell_counts',
    'tpi',
]

N_STATIC: int = len(STATIC_CHANNELS)


# ── Normalization ───────────────────────────────────────────────────

# dB normalization: clip to [-30, 5] then scale to roughly [0, 1]
DB_MIN = -30.0
DB_MAX = 5.0


def normalize_db(arr: np.ndarray) -> np.ndarray:
    """Normalize dB backscatter to [0, 1] range."""
    return np.clip((arr - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0).astype(np.float32)


def normalize_anf(arr: np.ndarray) -> np.ndarray:
    """ANF → [0, 1]. Higher ANF (worse quality) → lower output."""
    return (1.0 / (1.0 + np.log1p(arr))).astype(np.float32)


# Re-use static normalization from pairwise classifier
from sarvalanche.ml.pairwise_debris_classifier.channels import normalize_static_channel  # noqa: E402, F401
