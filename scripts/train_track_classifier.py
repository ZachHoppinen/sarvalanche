"""
Train an XGBoost track-level avalanche debris classifier.

Usage
-----
    conda run -n sarvalanche python scripts/train_track_classifier.py

Reads track_labels.json, extracts per-track raster features, reports 5-fold
cross-validated ROC-AUC, fits a final model on all labeled data, prints the
top feature importances, and saves the model to disk.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from sarvalanche.ml.track_classifier import (
    BINARY_THRESHOLD,
    TRACK_PREDICTOR_DIR,
    TRACK_PREDICTOR_MODEL,
    build_training_set,
    train_classifier,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s – %(message)s')
log = logging.getLogger(__name__)

LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
RUNS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')

# ── Load labels ───────────────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    labels = json.load(f)

log.info("Loaded %d labeled tracks", len(labels))

label_counts = {}
for v in labels.values():
    label_counts[v['label']] = label_counts.get(v['label'], 0) + 1
log.info("Raw label distribution: %s  (threshold for debris: >=%d)", label_counts, BINARY_THRESHOLD)

# ── Build feature matrix ──────────────────────────────────────────────────────
X, y = build_training_set(labels, RUNS_DIR)
log.info("Feature matrix: %d rows × %d cols", *X.shape)
log.info("Class balance:  %s", y.value_counts().to_dict())

# ── 5-fold cross-validation ───────────────────────────────────────────────────
neg, pos = int((y == 0).sum()), int((y == 1).sum())
clf_cv = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=(neg / pos) if neg > pos else 1.0,
    eval_metric='logloss',
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_filled = X.fillna(X.median())

for metric in ('roc_auc', 'average_precision'):
    scores = cross_val_score(clf_cv, X_filled, y, cv=cv, scoring=metric)
    log.info("5-fold CV %-20s %.3f ± %.3f", metric + ':', scores.mean(), scores.std())

# ── Train final model on all labeled data ─────────────────────────────────────
clf = train_classifier(X, y)

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(clf.feature_importances_, index=clf.feature_names_in_)
print("\nTop 15 feature importances (gain):")
print(importances.sort_values(ascending=False).head(15).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
TRACK_PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, TRACK_PREDICTOR_MODEL)
log.info("Model saved → %s", TRACK_PREDICTOR_MODEL)
