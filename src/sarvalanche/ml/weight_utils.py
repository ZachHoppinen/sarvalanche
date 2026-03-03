"""Centralized discovery of ML model weight files.

Each model family stores weights under ``ml/weights/<subdir>/``.
This module provides ``find_weights(model)`` to locate the latest
matching file by glob, so callers never hard-code a specific versioned
filename.

Supported models
----------------
- ``"rtc_predictor"``   → weights/rtc_predictor/sar_transformer_best*.pth
                          SAR Transformer for baseline RTC prediction.
- ``"track_classifier"``→ weights/track_predictor/track_classifier*.joblib
                          XGBoost track-level debris classifier.
- ``"debris_segmenter"``→ weights/cnn_debris_detector/seg_model_best*.pt
                          UNet++ pixel-wise debris segmentation CNN.
"""


from pathlib import Path

_WEIGHTS_ROOT = Path(__file__).parent / "weights"

_REGISTRY: dict[str, tuple[str, str]] = {
    "rtc_predictor":    ("rtc_predictor",      "sar_transformer_best*.pth"),
    "track_classifier": ("track_predictor",    "track_classifier*.joblib"),
    "debris_segmenter": ("cnn_debris_detector", "seg_model_best*.pt"),
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

def save_weights(model: str, suffix: str = 'best') -> Path:
    """Return the path to write a new weight file for *model*.

    Creates the weights subdirectory if it doesn't exist.

    Parameters
    ----------
    model : str
        Key from the registry.
    suffix : str
        File suffix — e.g. 'best', 'last', 'epoch_042'.

    Returns
    -------
    Path — the full path to write to (file not yet created).
    """
    if model not in _REGISTRY:
        raise KeyError(f"Unknown model {model!r}. Available: {sorted(_REGISTRY)}")
    subdir, default_pattern = _REGISTRY[model]
    weights_dir = _WEIGHTS_ROOT / subdir
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from pattern — strip glob chars, insert suffix
    stem = default_pattern.replace('*', f'_{suffix}').rstrip('.*')
    # Ensure correct extension
    ext = Path(default_pattern).suffix
    return weights_dir / f"{Path(stem).stem}{ext}"