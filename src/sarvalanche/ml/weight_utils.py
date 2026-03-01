"""Centralized discovery of ML model weight files.

Each model family stores weights under ``ml/weights/<subdir>/``.
This module provides ``find_weights(model, variant)`` to locate the
latest matching file by glob, so callers never hard-code a specific
versioned filename.

Supported models
----------------
- ``"rtc_predictor"``   → ``weights/rtc_predictor/sar_transformer_best*.pth``
- ``"cnn_seg_encoder"`` → ``weights/cnn_debris_detector/track_seg_encoder*.pt``
- ``"cnn_patch_encoder"`` → ``weights/cnn_debris_detector/track_patch_encoder*.pt``
- ``"track_classifier"`` → ``weights/track_predictor/track_classifier*.joblib``
"""

from pathlib import Path

_WEIGHTS_ROOT = Path(__file__).parent / "weights"

# (subdir, glob pattern) for each model family
_REGISTRY: dict[str, tuple[str, str]] = {
    "rtc_predictor":    ("rtc_predictor",      "sar_transformer_best*.pth"),
    "cnn_seg_encoder":  ("cnn_debris_detector", "track_seg_encoder.pt"),
    "cnn_patch_encoder":("cnn_debris_detector", "track_patch_encoder.pt"),
    "track_classifier": ("track_predictor",     "track_classifier*.joblib"),
}


def find_weights(model: str, pattern_override: str | None = None) -> Path:
    """Return the path to the latest weight file for *model*.

    Parameters
    ----------
    model : str
        Key from the registry (e.g. ``"rtc_predictor"``).
    pattern_override : str, optional
        Custom glob pattern to use instead of the registry default.

    Returns
    -------
    Path
        Absolute path to the weight file.

    Raises
    ------
    KeyError
        If *model* is not in the registry.
    FileNotFoundError
        If no matching file is found.
    """
    if model not in _REGISTRY:
        raise KeyError(
            f"Unknown model {model!r}. Available: {sorted(_REGISTRY)}"
        )
    subdir, default_pattern = _REGISTRY[model]
    pattern = pattern_override or default_pattern
    weights_dir = _WEIGHTS_ROOT / subdir
    candidates = sorted(weights_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} found in {weights_dir}"
        )
    # sorted() gives lexicographic order — last entry is the latest version
    return candidates[-1]
