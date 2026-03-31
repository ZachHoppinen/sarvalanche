"""Channel definitions for TinyCD-inspired debris classifier.

Like the Siamese model, feeds raw pre/post dB images and lets the
model learn change representations. Uses TinyCD's efficient architecture
with squeeze-excite attention and depthwise separable convolutions.

Per-branch channels (pre and post each get these):
  vv, vh, anf  (3)

Static (shared, injected at decoder):
  slope, aspect_northing, aspect_easting, cell_counts, tpi  (5)
"""

import numpy as np

# Re-use from pairwise
from sarvalanche.ml.pairwise_debris_classifier.channels import (  # noqa: F401
    STATIC_CHANNELS,
    N_STATIC,
    normalize_anf,
    normalize_static_channel,
)

BRANCH_CHANNELS: list[str] = [
    'vv',
    'vh',
    'anf',
]

N_BRANCH: int = len(BRANCH_CHANNELS)

# dB normalization for absolute backscatter
DB_MIN = -30.0
DB_MAX = 5.0


def normalize_db(arr: np.ndarray) -> np.ndarray:
    """Normalize dB backscatter to [0, 1] range."""
    return np.clip((arr - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0).astype(np.float32)
