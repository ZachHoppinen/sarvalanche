import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from sarvalanche.ml.track_classifier import (
    BINARY_THRESHOLD,
    TRACK_PREDICTOR_DIR,
    TRACK_PREDICTOR_MODEL,
    train_classifier,
    _zone_from_gpkg,
)


# ── Constants ───────────────────────────────────────────────────────────────


def test_binary_threshold():
    assert BINARY_THRESHOLD == 2


def test_predictor_paths():
    """Weight paths should point to src/sarvalanche/ml/weights/track_predictor/."""
    assert 'sarvalanche/ml/weights/track_predictor' in str(TRACK_PREDICTOR_DIR)
    assert str(TRACK_PREDICTOR_MODEL).endswith('track_classifier.joblib')


# ── _zone_from_gpkg ─────────────────────────────────────────────────────────


def test_zone_from_gpkg_basic():
    assert _zone_from_gpkg(Path('Zone_Name_2025-02-04.gpkg')) == 'Zone_Name'


def test_zone_from_gpkg_multiple_underscores():
    assert _zone_from_gpkg(Path('Some_Long_Zone_Name_2024-12-31.gpkg')) == 'Some_Long_Zone_Name'


def test_zone_from_gpkg_no_date():
    """Should return full stem if no date pattern found."""
    assert _zone_from_gpkg(Path('NoDateHere.gpkg')) == 'NoDateHere'


# ── train_classifier ────────────────────────────────────────────────────────


def test_train_classifier_basic():
    """Should fit an XGBoost model on a simple feature matrix."""
    np.random.seed(42)
    n = 30
    X = pd.DataFrame({
        'feat_a': np.random.rand(n),
        'feat_b': np.random.rand(n),
        'feat_c': np.random.rand(n),
    })
    y = pd.Series([0] * 15 + [1] * 15, name='debris')
    clf = train_classifier(X, y)
    assert hasattr(clf, 'predict')
    assert hasattr(clf, 'predict_proba')
    assert list(clf.feature_names_in_) == ['feat_a', 'feat_b', 'feat_c']


def test_train_classifier_handles_nan():
    """Classifier should handle NaN values via median imputation."""
    np.random.seed(42)
    n = 20
    X = pd.DataFrame({
        'a': np.random.rand(n),
        'b': np.random.rand(n),
    })
    X.iloc[0, 0] = np.nan
    X.iloc[5, 1] = np.nan
    y = pd.Series([0] * 10 + [1] * 10, name='debris')
    clf = train_classifier(X, y)
    proba = clf.predict_proba(X.fillna(X.median()))
    assert proba.shape == (n, 2)


def test_train_classifier_imbalanced():
    """scale_pos_weight should be set when negatives outnumber positives."""
    np.random.seed(42)
    X = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20)})
    y = pd.Series([0] * 18 + [1] * 2, name='debris')
    clf = train_classifier(X, y)
    assert clf.get_params()['scale_pos_weight'] == 9.0  # 18/2


def test_train_classifier_balanced():
    """scale_pos_weight should be 1.0 when classes are balanced."""
    np.random.seed(42)
    X = pd.DataFrame({'a': np.random.rand(10), 'b': np.random.rand(10)})
    y = pd.Series([0] * 5 + [1] * 5, name='debris')
    clf = train_classifier(X, y)
    assert clf.get_params()['scale_pos_weight'] == 1.0
