"""
full_season_manual_labeler.py — Manual track labeling tool for season datasets.

Works with the single season_dataset.nc + season_tracks.gpkg produced by the
dual-tau season runner.  Computes d_empirical on-the-fly for arbitrary dates
(tau=6 days) and rotates to a new random date every 50 labels.

Usage:
    conda run -n sarvalanche python scripts/manual_labeling/full_season_manual_labeler.py \
        --season-nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
        --tracks-gpkg local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_tracks.gpkg \
        --zone Sawtooth_&_Western_Smoky_Mtns
"""

import argparse
import ast
import json
import random
import re
import threading
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, LassoSelector
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Manual track labeling tool (season dataset)')
parser.add_argument('--season-nc', type=Path, required=True,
                    help='Path to season_dataset.nc')
parser.add_argument('--tracks-gpkg', type=Path, required=True,
                    help='Path to season_tracks.gpkg')
parser.add_argument('--zone', type=str, required=True,
                    help='Zone name (e.g. Sawtooth_&_Western_Smoky_Mtns)')
parser.add_argument('--random', action='store_true',
                    help='Skip XGBoost scoring, present tracks in random order')
parser.add_argument('--no-retrain', action='store_true',
                    help='Disable background retraining of XGBoost classifier')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OBS_PATH    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/snfac_obs_2021_2025.csv')
SHAPES_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')
TAU_DAYS    = 6
DATE_ROTATION_INTERVAL = 50

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

layers = {
    'd_empirical': {'cmap': 'RdBu_r',   'label': 'Empirical Distance',  'vmin': -2.0, 'vmax': 2.0,
                    'thresholds': [0.5, 1.0]},
    'slope':       {'cmap': 'bone',      'label': 'Slope (rad)',         'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45),
                    'thresholds': [np.deg2rad(30), np.deg2rad(38)]},
    'cell_counts': {'cmap': 'Blues',     'label': 'Cell Counts',         'vmin': None, 'vmax': None,
                    'thresholds': [5, 20]},
    'p_empirical': {'cmap': 'RdYlGn',   'label': 'P(empirical)',        'vmin': 0, 'vmax': 1,
                    'thresholds': [0.3, 0.7]},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track_stats(da, geom_gdf, thresholds):
    """Return (threshold, pct, count) tuples for pixels inside the track polygon."""
    try:
        clipped = da.rio.clip(geom_gdf.geometry.values, all_touched=True, drop=True)
        vals = clipped.values.ravel()
        vals = vals[~np.isnan(vals)]
        n_total = len(vals)
        if n_total == 0:
            return []
        return [(t, 100.0 * float((vals >= t).sum()) / n_total, int((vals >= t).sum()))
                for t in thresholds]
    except Exception:
        return []


def label_key(zone, date, track_id):
    return f"{zone}|{date}|{track_id}"


def _get_track_id(gdf, idx):
    """Get track ID from 'id' column, fall back to integer index."""
    if 'id' in gdf.columns:
        return str(gdf.loc[idx, 'id'])
    return str(idx)


# ---------------------------------------------------------------------------
# On-the-fly empirical computation
# ---------------------------------------------------------------------------

_cached_date = None
_cached_layers = {}  # {'d_empirical': DataArray, 'p_empirical': DataArray}


def compute_empirical_for_date(ds, reference_date):
    """Compute d_empirical and p_empirical for a given date, caching results."""
    global _cached_date, _cached_layers

    ref_ts = np.datetime64(reference_date)
    if _cached_date is not None and _cached_date == ref_ts:
        return _cached_layers

    from sarvalanche.weights.temporal import get_temporal_weights
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )

    print(f"[compute] Computing empirical layers for {reference_date} (tau={TAU_DAYS}d)...")

    # Clean stale variables
    stale_patterns = [
        re.compile(r"^p_\d+_V[VH]_empirical$"),
        re.compile(r"^d_\d+_V[VH]_empirical$"),
    ]
    stale_exact = {"p_empirical", "d_empirical", "w_temporal"}
    to_drop = [v for v in ds.data_vars if v in stale_exact]
    for pat in stale_patterns:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        for v in to_drop:
            del ds[v]

    # Temporal weights
    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=TAU_DAYS)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    # Empirical backscatter probability
    p_empirical, d_empirical = calculate_empirical_backscatter_probability(
        ds,
        ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )

    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical

    _cached_date = ref_ts
    _cached_layers = {'d_empirical': d_empirical, 'p_empirical': p_empirical}
    print(f"[compute] Done. d_empirical range: [{float(d_empirical.min()):.2f}, {float(d_empirical.max()):.2f}]")
    return _cached_layers


# ---------------------------------------------------------------------------
# Load season dataset + tracks
# ---------------------------------------------------------------------------

print(f"Loading season dataset: {args.season_nc}")
from sarvalanche.io.dataset import load_netcdf_to_dataset
ds = load_netcdf_to_dataset(args.season_nc)

# Ensure time is datetime
if not np.issubdtype(ds["time"].dtype, np.datetime64):
    ds["time"] = pd.DatetimeIndex(ds["time"].values)

# Load into memory for fast pixel access
if any(var.chunks is not None for var in ds.variables.values()):
    print("Loading dataset into memory...")
    ds = ds.load()

print(f"  {len(ds.time)} time steps, {ds.sizes['y']}×{ds.sizes['x']} spatial")

# Ensure w_resolution exists
if "w_resolution" not in ds.data_vars:
    raise RuntimeError(
        "w_resolution not found in season dataset — "
        "re-run the season pipeline to include resolution weights"
    )

print(f"Loading tracks: {args.tracks_gpkg}")
gdf = gpd.read_file(args.tracks_gpkg)
print(f"  {len(gdf)} tracks")

# Reproject dataset to match tracks CRS
if ds.rio.crs != gdf.crs:
    print(f"Reprojecting dataset from {ds.rio.crs} to {gdf.crs}...")
    ds = ds.rio.reproject(gdf.crs)

zone = args.zone
all_dates = pd.DatetimeIndex(ds["time"].values)

# ---------------------------------------------------------------------------
# Build all items: (zone, date_str, track_idx, track_id)
# ---------------------------------------------------------------------------

all_items = []
for date in all_dates:
    date_str = str(date.date())
    for idx in gdf.index:
        track_id = _get_track_id(gdf, idx)
        all_items.append((zone, date_str, idx, track_id))

print(f"{len(all_items)} total items ({len(all_dates)} dates × {len(gdf)} tracks)")

# ---------------------------------------------------------------------------
# Debris shapes
# ---------------------------------------------------------------------------

if SHAPES_PATH.exists():
    debris_shapes = gpd.read_file(SHAPES_PATH)
    print(f"Loaded {len(debris_shapes)} existing debris shapes from {SHAPES_PATH}")
else:
    debris_shapes = gpd.GeoDataFrame(
        columns=['key', 'zone', 'date', 'track_idx', 'geometry'],
        geometry='geometry',
    )


def _save_shapes():
    if debris_shapes.crs is None and not debris_shapes.empty:
        print("[shapes] Warning: no CRS set, skipping save")
        return
    debris_shapes.to_file(SHAPES_PATH, driver='GPKG')


def _get_existing_shapes(key, target_crs=None):
    if debris_shapes.empty:
        return debris_shapes
    subset = debris_shapes[debris_shapes['key'] == key]
    if not subset.empty and target_crs is not None and debris_shapes.crs is not None:
        subset = subset.to_crs(target_crs)
    return subset


# ---------------------------------------------------------------------------
# ShapeDrawer (identical to original)
# ---------------------------------------------------------------------------

class ShapeDrawer:
    """Interactive polygon / lasso drawing on a matplotlib axes."""

    def __init__(self, ax, data_axes, crs):
        self.ax = ax
        self.data_axes = list(data_axes)
        self._data_axes_set = set(self.data_axes)
        self.crs = crs
        self.shapes: list[Polygon] = []
        self._patches: list = []
        self.draw_mode = False
        self.tool_mode = 'polygon'
        self._selector = None
        self._status_text = None
        self._click_cid = ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_click,
        )

    def _on_click(self, event):
        if not self.draw_mode or event.inaxes not in self._data_axes_set:
            return
        if event.inaxes is self.ax:
            return
        self.ax = event.inaxes
        self._activate_selector()
        self._update_status()

    def _update_status(self):
        if self._status_text is not None:
            self._status_text.remove()
            self._status_text = None
        if self.draw_mode:
            mode_str = f"DRAW: {self.tool_mode}  (d=exit  p=polygon  l=lasso  z=undo)"
            self._status_text = self.ax.text(
                0.5, 0.02, mode_str, transform=self.ax.transAxes,
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9),
                zorder=100,
            )
        self.ax.figure.canvas.draw_idle()

    def _activate_selector(self):
        if self._selector is not None:
            self._selector.disconnect_events()
            self._selector = None
        if not self.draw_mode:
            return
        if self.tool_mode == 'polygon':
            self._selector = PolygonSelector(
                self.ax, self._on_polygon_complete,
                useblit=True,
                props=dict(color='red', linewidth=2, alpha=0.7),
                handle_props=dict(markersize=5),
            )
        else:
            self._selector = LassoSelector(
                self.ax, self._on_lasso_complete,
                useblit=True,
                props=dict(color='red', linewidth=2, alpha=0.7),
            )

    def _on_polygon_complete(self, verts):
        if len(verts) < 3:
            return
        poly = Polygon(verts)
        if poly.is_valid and not poly.is_empty:
            self.shapes.append(poly)
            self._draw_shape(poly)
            print(f"  [shape] Polygon drawn ({len(verts)} vertices, {poly.area:.0f} m²)")

    def _on_lasso_complete(self, verts):
        if len(verts) < 3:
            return
        poly = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return
        self.shapes.append(poly)
        self._draw_shape(poly)
        print(f"  [shape] Lasso drawn ({len(verts)} vertices, {poly.area:.0f} m²)")

    def _draw_shape(self, poly):
        from matplotlib.patches import Polygon as MplPolygon
        for ax in self.data_axes:
            p = MplPolygon(
                list(poly.exterior.coords), closed=True,
                fill=True, facecolor='red', edgecolor='red',
                alpha=0.3, linewidth=2, zorder=10,
            )
            ax.add_patch(p)
            self._patches.append(p)
        self.ax.figure.canvas.draw_idle()

    def undo(self):
        if not self.shapes:
            print("  [shape] Nothing to undo")
            return
        self.shapes.pop()
        n_axes = len(self.data_axes)
        for _ in range(min(n_axes, len(self._patches))):
            if self._patches:
                self._patches.pop().remove()
        self.ax.figure.canvas.draw_idle()
        print("  [shape] Undone last shape")

    def toggle_draw(self):
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self._activate_selector()
        else:
            if self._selector is not None:
                self._selector.disconnect_events()
                self._selector = None
        self._update_status()

    def set_tool(self, mode):
        if mode in ('polygon', 'lasso'):
            self.tool_mode = mode
            if self.draw_mode:
                self._activate_selector()
            self._update_status()

    def plot_existing(self, existing_gdf):
        from matplotlib.patches import Polygon as MplPolygon
        for _, row in existing_gdf.iterrows():
            poly = row.geometry
            if poly is None or poly.is_empty:
                continue
            for ax in self.data_axes:
                p = MplPolygon(
                    list(poly.exterior.coords), closed=True,
                    fill=True, facecolor='lime', edgecolor='lime',
                    alpha=0.25, linewidth=1.5, linestyle='--', zorder=9,
                )
                ax.add_patch(p)


# ---------------------------------------------------------------------------
# XGBoost scoring (optional)
# ---------------------------------------------------------------------------

try:
    from sarvalanche.ml.track_classifier import (
        TRACK_PREDICTOR_DIR,
        TRACK_PREDICTOR_MODEL,
        build_training_set,
        train_classifier,
    )
    from sarvalanche.ml.track_features import (
        compute_scene_context,
        extract_track_features,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier

    clf = None
    if TRACK_PREDICTOR_MODEL.exists():
        clf = joblib.load(TRACK_PREDICTOR_MODEL)
        print(f"Classifier loaded from {TRACK_PREDICTOR_MODEL}")
    else:
        print(f"No classifier found at {TRACK_PREDICTOR_MODEL}, predictions disabled")
except ImportError as e:
    print(f"ML modules not available ({e}), scoring disabled")
    clf = None


SCORE_CACHE_PATH = args.season_nc.parent / '_track_scores_season.json'


def _load_score_cache():
    if not SCORE_CACHE_PATH.exists():
        return {}
    try:
        with open(SCORE_CACHE_PATH) as f:
            raw = json.load(f)
        return {
            (parts[0], parts[1], parts[2]): v
            for k, v in raw.items()
            if len(parts := k.split('|')) == 3
        }
    except Exception as exc:
        print(f"[score] Failed to load cache: {exc}")
        return {}


def _save_score_cache(scores):
    try:
        raw = {f"{z}|{d}|{tid}": p for (z, d, tid), p in scores.items()}
        with open(SCORE_CACHE_PATH, 'w') as f:
            json.dump(raw, f)
    except Exception as exc:
        print(f"[score] Failed to save cache: {exc}")


def _score_unlabeled_tracks(items, classifier, ds_for_scoring, gdf_for_scoring, use_cache=True):
    """Score unlabeled tracks with XGBoost."""
    scores = {}
    if classifier is None:
        return scores

    cached = _load_score_cache() if use_cache else {}
    needed = []
    for item in items:
        key = (item[0], item[1], item[3])  # zone, date, track_id
        if key in cached:
            scores[key] = cached[key]
        else:
            needed.append(item)

    if cached and scores:
        print(f"[score] Loaded {len(scores)} cached scores, {len(needed)} need scoring")

    if not needed:
        return scores

    # Group by date to compute features once per date
    date_groups = defaultdict(list)
    for zone, date, idx, track_id in needed:
        date_groups[date].append((zone, date, idx, track_id))

    n_scored = 0
    n_failed = 0
    for di, (date_str, track_list) in enumerate(date_groups.items()):
        print(f"[score] Date {di+1}/{len(date_groups)}: {date_str} ({len(track_list)} tracks)")
        try:
            # Compute empirical for this date
            compute_empirical_for_date(ds_for_scoring, pd.Timestamp(date_str))
            scene_ctx = compute_scene_context(ds_for_scoring)

            for zone, date, idx, track_id in track_list:
                try:
                    row = gdf_for_scoring.loc[idx]
                    feats = extract_track_features(row, ds_for_scoring, scene_ctx=scene_ctx)
                    X_row = pd.DataFrame([feats]).reindex(columns=classifier.feature_names_in_).fillna(0)
                    p = float(classifier.predict_proba(X_row)[0, 1])
                    scores[(zone, date, track_id)] = p
                    n_scored += 1
                except Exception:
                    n_failed += 1
        except Exception as exc:
            print(f"[score] Failed for date {date_str}: {exc}")
            n_failed += len(track_list)

    print(f"[score] Scored {n_scored} tracks, {n_failed} failed")
    _save_score_cache(scores)
    return scores


def _sort_by_uncertainty(items, scores):
    """Sort unlabeled items using active learning strategy."""
    uncertain = []
    tail = []
    middle = []
    unscored = []

    for item in items:
        key = (item[0], item[1], item[3])
        if key not in scores:
            unscored.append(item)
            continue
        p = scores[key]
        dist = abs(p - 0.5)
        if 0.25 <= p <= 0.75:
            uncertain.append((dist, item))
        elif p < 0.15 or p > 0.85:
            tail.append((dist, item))
        else:
            middle.append((dist, item))

    uncertain.sort(key=lambda x: x[0])
    uncertain_items = [it for _, it in uncertain]
    random.shuffle(tail)
    tail_items = [it for _, it in tail]
    middle_items = [it for _, it in middle]
    random_pool = middle_items + list(uncertain_items) + list(tail_items)
    random.shuffle(random_pool)
    random.shuffle(unscored)

    result = []
    ui, ri, ti, si = 0, 0, 0, 0
    cycle = 0
    total = len(items)

    while len(result) < total:
        slot = cycle % 10
        if slot < 6 and ui < len(uncertain_items):
            result.append((uncertain_items[ui], 'uncertain'))
            ui += 1
        elif slot < 8 and ri < len(random_pool):
            result.append((random_pool[ri], 'random'))
            ri += 1
        elif slot < 9 and ti < len(tail_items):
            result.append((tail_items[ti], 'tail'))
            ti += 1
        elif si < len(unscored):
            result.append((unscored[si], 'unscored'))
            si += 1
        else:
            if ui < len(uncertain_items):
                result.append((uncertain_items[ui], 'uncertain'))
                ui += 1
            elif ri < len(random_pool):
                result.append((random_pool[ri], 'random'))
                ri += 1
            elif ti < len(tail_items):
                result.append((tail_items[ti], 'tail'))
                ti += 1
            elif si < len(unscored):
                result.append((unscored[si], 'unscored'))
                si += 1
            else:
                break
        cycle += 1

    seen = set()
    deduped = []
    for item, tag in result:
        key = (item[0], item[1], item[3])
        if key not in seen:
            seen.add(key)
            deduped.append((item, tag))

    n_unc = sum(1 for _, t in deduped if t == 'uncertain')
    n_rnd = sum(1 for _, t in deduped if t == 'random')
    n_tail = sum(1 for _, t in deduped if t == 'tail')
    n_uns = sum(1 for _, t in deduped if t == 'unscored')
    print(f"[sort] Ordering: {n_unc} uncertain, {n_rnd} random, {n_tail} tail, {n_uns} unscored")
    return deduped


# ---------------------------------------------------------------------------
# Track area cache + size bias
# ---------------------------------------------------------------------------

_area_cache: dict[int, float] = {
    i: geom.area for i, geom in zip(gdf.index, gdf.geometry) if geom is not None
}


def _bias_towards_large(items, large_frac=0.8):
    """Reorder items so ~80% are from the larger half."""
    areas = [_area_cache.get(item[2], 0.0) for item in items]
    if not areas:
        return items
    median_area = float(np.median(areas))

    large = [item for item, a in zip(items, areas) if a >= median_area]
    small = [item for item, a in zip(items, areas) if a < median_area]

    result = []
    li, si = 0, 0
    cycle = 0
    while li < len(large) or si < len(small):
        if cycle % 5 < 4 and li < len(large):
            result.append(large[li])
            li += 1
        elif si < len(small):
            result.append(small[si])
            si += 1
        elif li < len(large):
            result.append(large[li])
            li += 1
        else:
            break
        cycle += 1
    return result


# ---------------------------------------------------------------------------
# Background retrain
# ---------------------------------------------------------------------------

_rescore_needed = False


def _retrain_in_background():
    global clf, _rescore_needed
    print(f"\n[retrain] Starting on {len(labels)} labels...")
    try:
        # Build training set from the season dataset directly
        # For now, use the existing build_training_set with RUNS_DIRS fallback
        # This is a simplified retrain that updates clf in-place
        print("[retrain] Background retrain not yet implemented for season datasets")
    except Exception as exc:
        print(f"[retrain] Failed: {exc}")


# ---------------------------------------------------------------------------
# Load observation CSV
# ---------------------------------------------------------------------------

if OBS_PATH.exists():
    _obs_raw = pd.read_csv(OBS_PATH)
    _obs_raw['date'] = pd.to_datetime(_obs_raw['date'])
    _obs_raw = _obs_raw.dropna(subset=['location_point'])
    _obs_raw['geometry'] = _obs_raw['location_point'].apply(
        lambda s: Point(ast.literal_eval(s)['lng'], ast.literal_eval(s)['lat'])
    )
    obs_gdf = gpd.GeoDataFrame(_obs_raw, geometry='geometry', crs='EPSG:4326')
else:
    obs_gdf = gpd.GeoDataFrame(columns=['date', 'geometry'], geometry='geometry', crs='EPSG:4326')

# ---------------------------------------------------------------------------
# Load existing labels
# ---------------------------------------------------------------------------

if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

# ---------------------------------------------------------------------------
# Date selection: random date, rotating every DATE_ROTATION_INTERVAL labels
# ---------------------------------------------------------------------------

_current_date = None
_labels_since_rotation = 0


def _pick_random_date():
    global _current_date, _labels_since_rotation
    _current_date = str(pd.Timestamp(np.random.choice(all_dates)).date())
    _labels_since_rotation = 0
    print(f"\n>>> New random date: {_current_date} <<<\n")
    return _current_date


def _maybe_rotate_date():
    global _labels_since_rotation
    _labels_since_rotation += 1
    if _labels_since_rotation >= DATE_ROTATION_INTERVAL:
        _pick_random_date()


# Pick initial date
_pick_random_date()


# ---------------------------------------------------------------------------
# Build items for the current date
# ---------------------------------------------------------------------------

def _items_for_date(date_str):
    """Build (zone, date_str, idx, track_id) list for one date."""
    items = []
    for idx in gdf.index:
        track_id = _get_track_id(gdf, idx)
        key = label_key(zone, date_str, track_id)
        if key not in labels:
            items.append((zone, date_str, idx, track_id))
    return items


def _get_unlabeled_for_current_date():
    """Get unlabeled items for the current date, scored and sorted."""
    items = _items_for_date(_current_date)
    if not items:
        return [], []

    if args.random or clf is None:
        random.shuffle(items)
        items = _bias_towards_large(items)
        return items, ['random'] * len(items)

    scores = _score_unlabeled_tracks(items, clf, ds, gdf)
    if scores:
        sorted_items = _sort_by_uncertainty(items, scores)
        reordered = _bias_towards_large([item for item, _ in sorted_items])
        tag_lookup = {(it[0], it[1], it[3]): tag for it, tag in sorted_items}
        items_out = reordered
        tags_out = [tag_lookup.get((it[0], it[1], it[3]), 'random') for it in reordered]
        return items_out, tags_out

    random.shuffle(items)
    items = _bias_towards_large(items)
    return items, ['random'] * len(items)


# Filter already labeled, count
label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for v in labels.values():
    lbl = v['label']
    if lbl in label_counts:
        label_counts[lbl] += 1
n_pos = label_counts[2] + label_counts[3]
n_neg = label_counts[0] + label_counts[1]
print(f"{len(labels)} labeled ({n_neg} neg [{label_counts[0]}×0, {label_counts[1]}×1] / "
      f"{n_pos} pos [{label_counts[2]}×2, {label_counts[3]}×3])")

# Initial batch
unlabeled, unlabeled_tags = _get_unlabeled_for_current_date()
print(f"{len(unlabeled)} unlabeled tracks for date {_current_date}")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

result = {'label': None}
active_drawer: ShapeDrawer | None = None


def on_key(event):
    global active_drawer
    if event.key == 'd':
        if active_drawer is not None:
            active_drawer.toggle_draw()
        return
    if active_drawer is not None and active_drawer.draw_mode:
        if event.key == 'p':
            active_drawer.set_tool('polygon')
            return
        elif event.key == 'l':
            active_drawer.set_tool('lasso')
            return
        elif event.key == 'z':
            active_drawer.undo()
            return
        if event.key == 'q':
            result['label'] = 'quit'
            plt.close()
        return

    if event.key in ['0', '1', '2', '3']:
        result['label'] = int(event.key)
        plt.close()
    elif event.key == 'n':
        result['label'] = -1
        plt.close()
    elif event.key == 'q':
        result['label'] = 'quit'
        plt.close()
    elif event.key == 'left':
        result['label'] = 'back'
        plt.close()
    elif event.key == 'right':
        result['label'] = 'next'
        plt.close()


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

buffer = 500
i = 0
quit_requested = False

while not quit_requested:
    # Refill if exhausted or date rotated
    while i >= len(unlabeled):
        _pick_random_date()
        unlabeled, unlabeled_tags = _get_unlabeled_for_current_date()
        i = 0
        if not unlabeled:
            print(f"No unlabeled tracks for date {_current_date}, trying another...")
            continue
        print(f"{len(unlabeled)} unlabeled tracks for date {_current_date}")

    zone_i, date_i, idx_i, track_id_i = unlabeled[i]

    # Compute empirical layers for this date (cached)
    try:
        compute_empirical_for_date(ds, pd.Timestamp(date_i))
    except Exception as exc:
        print(f"[error] Failed to compute empirical for {date_i}: {exc}")
        i += 1
        continue

    row = gdf.loc[idx_i]
    bounds = row.geometry.bounds
    path_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)

    clip_kwargs = dict(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer),
    )

    key = label_key(zone_i, date_i, track_id_i)
    existing = labels.get(key)
    existing_str = f"  [labeled: {existing['label']}]" if existing else ""

    avl_date = pd.Timestamp(date_i)
    nearby_obs = obs_gdf[
        (obs_gdf['date'] - avl_date).abs() <= pd.Timedelta(days=6)
    ]
    if not nearby_obs.empty:
        nearby_obs = nearby_obs.to_crs(gdf.crs)
    obs_str = f"  ({len(nearby_obs)} obs ±6d)" if not nearby_obs.empty else ""

    existing_shapes = _get_existing_shapes(key, target_crs=gdf.crs)
    shape_str = f"  [{len(existing_shapes)} shapes]" if len(existing_shapes) > 0 else ""

    bucket_tag = unlabeled_tags[i] if i < len(unlabeled_tags) else 'random'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{zone_i}  |  {date_i}  |  {track_id_i}{existing_str}{obs_str}{shape_str}  "
        f"({i + 1}/{len(unlabeled)})  [{bucket_tag}]\n"
        f"0-3=label  n=unsure  \u2190/\u2192=nav  d=draw  q=quit",
        fontsize=11,
    )
    fig.canvas.mpl_connect('key_press_event', on_key)

    active_drawer = ShapeDrawer(axes.flat[0], axes.flat, gdf.crs)
    if not existing_shapes.empty:
        active_drawer.plot_existing(existing_shapes)

    for ax, (var, opts) in zip(axes.flat, layers.items()):
        plot_kwargs = dict(ax=ax, cmap=opts['cmap'], robust=True, add_colorbar=True)
        if opts['vmin'] is not None:
            plot_kwargs['vmin'] = opts['vmin']
        if opts['vmax'] is not None:
            plot_kwargs['vmax'] = opts['vmax']
        try:
            ds[var].sel(**clip_kwargs).plot(**plot_kwargs)
        except Exception:
            # Fallback: plot full extent
            ds[var].plot(**plot_kwargs)
        path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
        if not nearby_obs.empty:
            nearby_obs.plot(ax=ax, color='cyan', markersize=6, marker='o',
                            edgecolors='black', linewidths=0.5, zorder=5)

        title = opts['label']
        if opts.get('thresholds'):
            stats = _track_stats(ds[var], path_gdf, opts['thresholds'])
            if stats:
                parts = [f">{t:.2g}: {pct:.0f}% ({n}px)" for t, pct, n in stats]
                title += "\n" + "   ".join(parts)
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')

    plt.tight_layout()
    result['label'] = None
    plt.show()

    # Save drawn shapes
    if active_drawer is not None and active_drawer.shapes:
        new_rows = gpd.GeoDataFrame(
            [{'key': key, 'zone': zone_i, 'date': date_i, 'track_idx': track_id_i,
              'geometry': shape}
             for shape in active_drawer.shapes],
            geometry='geometry',
            crs=gdf.crs,
        )
        debris_shapes = pd.concat([debris_shapes, new_rows], ignore_index=True)
        debris_shapes = gpd.GeoDataFrame(debris_shapes, geometry='geometry', crs=gdf.crs)
        _save_shapes()
        print(f"  [shape] Saved {len(active_drawer.shapes)} shape(s) for {key}  "
              f"({len(debris_shapes)} total)")
    active_drawer = None

    if result['label'] == 'quit':
        quit_requested = True
    elif result['label'] == 'back':
        i = max(0, i - 1)
    elif result['label'] == 'next':
        i += 1
    elif result['label'] is not None:
        labels[key] = {
            'label': result['label'],
            'zone': zone_i,
            'date': date_i,
            'track_idx': track_id_i,
        }
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(labels, f, indent=2)
        label_counts[result['label']] = label_counts.get(result['label'], 0) + 1
        n_pos = label_counts[2] + label_counts[3]
        n_neg = label_counts[0] + label_counts[1]
        print(f"{zone_i} | {date_i} | {track_id_i} → {result['label']}  "
              f"({len(labels)} labeled: {n_neg} neg / {n_pos} pos)")

        _maybe_rotate_date()
        # If date rotated, refill
        if _labels_since_rotation == 0:
            unlabeled, unlabeled_tags = _get_unlabeled_for_current_date()
            i = 0
        else:
            i += 1

        if len(labels) % 50 == 0 and not args.no_retrain:
            threading.Thread(target=_retrain_in_background, daemon=True).start()

print(f"Done. {len(labels)} tracks labeled, saved to {OUTPUT_PATH}")
