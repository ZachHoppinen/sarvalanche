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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

import torch

from sarvalanche.ml.track_classifier import (
    TRACK_PREDICTOR_DIR,
    TRACK_PREDICTOR_MODEL,
    build_patch_training_set,
    build_training_set,
    train_classifier,
)
from sarvalanche.ml.track_features import (
    aggregate_seg_features,
    extract_track_features,
    extract_track_patch,
)
from sarvalanche.ml.track_patch_encoder import CNN_SEG_ENCODER_PATH, TrackSegEncoder
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')

RUNS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/sarvalanche_runs')
OUTPUT_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')
OBS_PATH    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/snfac_obs_2021_2025.csv')
SHAPES_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')

layers = {
    'distance_mahalanobis': {'cmap': 'plasma',   'label': 'Mahalanobis Distance', 'vmin': 0.2, 'vmax': 0.7,  'thresholds': [0.3, 0.5]},
    'p_empirical':          {'cmap': 'RdYlGn_r', 'label': 'Empirical p-value',    'vmin': 0.2, 'vmax': 0.7,  'thresholds': [0.3, 0.5]},
    'slope':                {'cmap': 'bone',      'label': 'Slope (rad)',           'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45), 'thresholds': [np.deg2rad(30), np.deg2rad(38)]},
    'cell_counts':          {'cmap': 'Blues',     'label': 'Cell Counts',           'vmin': None, 'vmax': None, 'thresholds': [5, 20]},
}


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


def parse_stem(stem):
    """Return (zone, date) from a filename stem like 'Banner_Summit_2025-02-04'."""
    m = re.search(r'^(.+)_(\d{4}-\d{2}-\d{2})$', stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def label_key(zone, date, track_idx):
    return f"{zone}|{date}|{track_idx}"


# ── Discover all (zone, date, gpkg, nc) combos ───────────────────────────────
all_items = []  # list of (zone, date, track_idx, gpkg_path, nc_path)
for gpkg in sorted(RUNS_DIR.glob('*.gpkg')):
    zone, date = parse_stem(gpkg.stem)
    if zone is None:
        print(f"Warning: could not parse zone/date from {gpkg.name}, skipping")
        continue
    nc = gpkg.with_suffix('.nc')
    if not nc.exists():
        print(f"Warning: no matching .nc for {gpkg.name}, skipping")
        continue
    gdf = gpd.read_file(gpkg)
    for idx in gdf.index:
        all_items.append((zone, date, idx, gpkg, nc))

# ── Load XGBoost classifier (optional) ───────────────────────────────────────
clf = None
if TRACK_PREDICTOR_MODEL.exists():
    clf = joblib.load(TRACK_PREDICTOR_MODEL)
    print(f"Classifier loaded from {TRACK_PREDICTOR_MODEL}")
else:
    print(f"No classifier found at {TRACK_PREDICTOR_MODEL}, predictions disabled")

# ── Load CNN seg encoder (optional) ──────────────────────────────────────────
seg_enc = None
if CNN_SEG_ENCODER_PATH.exists():
    seg_enc = TrackSegEncoder()
    seg_enc.load_state_dict(torch.load(CNN_SEG_ENCODER_PATH, weights_only=True))
    seg_enc.eval()
    print(f"Seg encoder loaded from {CNN_SEG_ENCODER_PATH}")
else:
    print(f"No seg encoder found at {CNN_SEG_ENCODER_PATH}, seg predictions disabled")

# ── Load existing debris shapes ──────────────────────────────────────────────
if SHAPES_PATH.exists():
    debris_shapes = gpd.read_file(SHAPES_PATH)
    print(f"Loaded {len(debris_shapes)} existing debris shapes from {SHAPES_PATH}")
else:
    debris_shapes = gpd.GeoDataFrame(
        columns=['key', 'zone', 'date', 'track_idx', 'geometry'],
        geometry='geometry',
    )


def _save_shapes():
    """Persist debris_shapes GeoDataFrame to disk."""
    if debris_shapes.crs is None and not debris_shapes.empty:
        # Should not happen, but guard against it
        print("[shapes] Warning: no CRS set on debris_shapes, skipping save")
        return
    debris_shapes.to_file(SHAPES_PATH, driver='GPKG')


def _get_existing_shapes(key: str, target_crs=None) -> gpd.GeoDataFrame:
    """Return subset of debris_shapes matching a label key, reprojected if needed."""
    if debris_shapes.empty:
        return debris_shapes
    subset = debris_shapes[debris_shapes['key'] == key]
    if not subset.empty and target_crs is not None and debris_shapes.crs is not None:
        subset = subset.to_crs(target_crs)
    return subset


class ShapeDrawer:
    """
    Interactive polygon / lasso drawing on a matplotlib axes.

    Press 'd' to toggle draw mode. In draw mode:
    - Default is polygon mode: click vertices, double-click to close.
    - Press 'l' to switch to lasso (freehand) mode, 'p' for polygon mode.
    - Press 'z' to undo the last drawn shape.

    Drawn shapes are stored as Shapely Polygons in plot (CRS) coordinates.
    """

    def __init__(self, ax, data_axes, crs):
        self.ax = ax
        self.data_axes = list(data_axes)  # only the 4 data panels, NOT colorbars
        self._data_axes_set = set(self.data_axes)
        self.crs = crs
        self.shapes: list[Polygon] = []
        self._patches: list = []  # matplotlib artists for drawn shapes
        self.draw_mode = False
        self.tool_mode = 'polygon'  # 'polygon' or 'lasso'
        self._selector = None
        self._status_text = None
        # Click handler to switch active drawing axes
        self._click_cid = ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_click,
        )

    def _on_click(self, event):
        """Switch the active drawing axes to whichever data panel was clicked."""
        if not self.draw_mode or event.inaxes not in self._data_axes_set:
            return
        if event.inaxes is self.ax:
            return  # already on this axes
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
        """Create the appropriate selector widget."""
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
        """Called when PolygonSelector completes."""
        if len(verts) < 3:
            return
        poly = Polygon(verts)
        if poly.is_valid and not poly.is_empty:
            self.shapes.append(poly)
            self._draw_shape(poly)
            print(f"  [shape] Polygon drawn ({len(verts)} vertices, {poly.area:.0f} m²)")

    def _on_lasso_complete(self, verts):
        """Called when LassoSelector completes."""
        if len(verts) < 3:
            return
        poly = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersections from freehand
        if poly.is_empty:
            return
        self.shapes.append(poly)
        self._draw_shape(poly)
        print(f"  [shape] Lasso drawn ({len(verts)} vertices, {poly.area:.0f} m²)")

    def _draw_shape(self, poly):
        """Render a shape on all data axes (not colorbars)."""
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
        """Remove the last drawn shape."""
        if not self.shapes:
            print("  [shape] Nothing to undo")
            return
        self.shapes.pop()
        # Each shape adds one patch per data axes
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

    def set_tool(self, mode: str):
        if mode in ('polygon', 'lasso'):
            self.tool_mode = mode
            if self.draw_mode:
                self._activate_selector()
            self._update_status()

    def plot_existing(self, existing_gdf: gpd.GeoDataFrame):
        """Plot previously saved shapes on all data axes."""
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


def _retrain_in_background():
    """Retrain XGBoost classifier on current labels and update the global clf.

    Only retrains the XGBoost model using aggregate track features.
    Skips CNN seg encoder inference to avoid excessive memory usage.
    """
    global clf
    print(f"\n[retrain] Starting on {len(labels)} labels (XGBoost only, no CNN)...")
    try:
        X, y = build_training_set(labels, RUNS_DIR)

        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        scale_pos_weight = (neg / pos) if pos > 0 and neg > pos else 1.0

        # 5-fold stratified CV for honest performance estimate
        cv_clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            random_state=42, verbosity=0,
        )
        X_filled = X.fillna(X.median())
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc = cross_val_score(cv_clf, X_filled, y, cv=cv, scoring='roc_auc')
        f1  = cross_val_score(cv_clf, X_filled, y, cv=cv, scoring='f1')

        new_clf = train_classifier(X, y)
        TRACK_PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_clf, TRACK_PREDICTOR_MODEL)
        clf = new_clf
        print(
            f"[retrain] Done — {len(labels)} labels  ({pos} pos / {neg} neg)\n"
            f"          5-fold CV:  AUC={auc.mean():.3f} ± {auc.std():.3f}"
            f"   F1={f1.mean():.3f} ± {f1.std():.3f}"
        )
    except Exception as exc:
        print(f"[retrain] Failed: {exc}")

# ── Load observation CSV ──────────────────────────────────────────────────────
_obs_raw = pd.read_csv(OBS_PATH)
_obs_raw['date'] = pd.to_datetime(_obs_raw['date'])
_obs_raw = _obs_raw.dropna(subset=['location_point'])
_obs_raw['geometry'] = _obs_raw['location_point'].apply(
    lambda s: Point(ast.literal_eval(s)['lng'], ast.literal_eval(s)['lat'])
)
obs_gdf = gpd.GeoDataFrame(_obs_raw, geometry='geometry', crs='EPSG:4326')

# ── Load existing labels ──────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        labels = json.load(f)
else:
    labels = {}

# ── Filter already labeled, then shuffle within each file ────────────────────
unlabeled_all = [
    item for item in all_items
    if label_key(item[0], item[1], item[2]) not in labels
]

# Group by nc_path so reproject only fires once per file, not once per track
groups = defaultdict(list)
for item in unlabeled_all:
    groups[item[4]].append(item)

group_keys = list(groups.keys())
random.shuffle(group_keys)
for key in group_keys:
    random.shuffle(groups[key])
unlabeled = [item for key in group_keys for item in groups[key]]

print(f"{len(labels)} already labeled, {len(unlabeled_all)} remaining out of {len(all_items)} total")

# ── UI ───────────────────────────────────────────────────────────────────────
result = {'label': None}
active_drawer: ShapeDrawer | None = None


def on_key(event):
    global active_drawer
    # Draw-mode keys: intercept before label keys
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
        # In draw mode, only allow quit/nav — block label keys so clicks work
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


# ── Main labeling loop ────────────────────────────────────────────────────────
buffer = 500
current_nc_path = None
ds = None
gdf = None

i = 0
while i < len(unlabeled):
    zone, date, idx, gpkg_path, nc_path = unlabeled[i]

    # Reload ds and gdf only when the source file changes
    if nc_path != current_nc_path:
        if ds is not None:
            ds.close()
        gdf = gpd.read_file(gpkg_path)
        _peek = xr.open_dataset(nc_path)
        _load_vars = list({
            *layers.keys(),
            *[v for v in _peek.data_vars if _peek[v].dims == ('y', 'x')],
        })
        _peek.close()
        ds = xr.open_dataset(nc_path)[_load_vars].astype(float).rio.write_crs('EPSG:4326')
        ds = ds.rio.reproject(gdf.crs)
        current_nc_path = nc_path

    path = gdf.loc[idx]
    bounds = path.geometry.bounds
    path_gdf = gpd.GeoDataFrame([path], geometry='geometry', crs=gdf.crs)

    clip_kwargs = dict(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer),
    )

    existing = labels.get(label_key(zone, date, idx))
    existing_str = f"  [labeled: {existing['label']}]" if existing else ""

    pred_str = ""
    if clf is not None:
        feats = extract_track_features(path, ds)
        # If clf was trained with seg features, append them now
        if seg_enc is not None and any(c.startswith('seg_') for c in clf.feature_names_in_):
            patch = extract_track_patch(path, ds)
            with torch.no_grad():
                seg_logits = seg_enc.segment(torch.FloatTensor(patch[np.newaxis]))
                seg_probs = torch.sigmoid(seg_logits).numpy()[0, 0]
            feats.update(aggregate_seg_features(seg_probs, patch[7]))
        X_row = pd.DataFrame([feats]).reindex(columns=clf.feature_names_in_).fillna(0)
        p_debris = float(clf.predict_proba(X_row)[0, 1])
        pred_str = f"  ▶ XGB: {p_debris:.0%} debris"

    if seg_enc is not None:
        patch = extract_track_patch(path, ds)
        with torch.no_grad():
            seg_logits = seg_enc.segment(torch.FloatTensor(patch[np.newaxis]))
            seg_probs = torch.sigmoid(seg_logits).numpy()[0, 0]
        seg_mean = float(seg_probs[patch[7] > 0.5].mean()) if (patch[7] > 0.5).any() else 0.0
        pred_str += f"  Seg: {seg_mean:.0%}"

    avl_date = pd.Timestamp(date)
    nearby_obs = obs_gdf[
        (obs_gdf['date'] - avl_date).abs() <= pd.Timedelta(days=6)
    ]
    if not nearby_obs.empty:
        nearby_obs = nearby_obs.to_crs(gdf.crs)
    obs_str = f"  ({len(nearby_obs)} obs ±6d)" if not nearby_obs.empty else ""

    key = label_key(zone, date, idx)
    existing_shapes = _get_existing_shapes(key, target_crs=gdf.crs)
    shape_str = f"  [{len(existing_shapes)} shapes]" if len(existing_shapes) > 0 else ""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{zone}  |  {date}  |  track {idx}{existing_str}{obs_str}{shape_str}  ({i + 1}/{len(unlabeled)}){pred_str}\n"
        f"0-3=label  n=unsure  ←/→=nav  d=draw  q=quit",
        fontsize=11,
    )
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Set up shape drawer on first (top-left) axes; pass only data axes, not colorbars
    active_drawer = ShapeDrawer(axes.flat[0], axes.flat, gdf.crs)
    if not existing_shapes.empty:
        active_drawer.plot_existing(existing_shapes)

    for ax, (var, opts) in zip(axes.flat, layers.items()):
        plot_kwargs = dict(ax=ax, cmap=opts['cmap'], robust=True, add_colorbar=True)
        if opts['vmin'] is not None:
            plot_kwargs['vmin'] = opts['vmin']
        if opts['vmax'] is not None:
            plot_kwargs['vmax'] = opts['vmax']
        ds[var].sel(**clip_kwargs).plot(**plot_kwargs)
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

    # Save any drawn shapes regardless of label action
    if active_drawer is not None and active_drawer.shapes:
        key = label_key(zone, date, idx)
        new_rows = gpd.GeoDataFrame(
            [{'key': key, 'zone': zone, 'date': date, 'track_idx': int(idx),
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
        break
    elif result['label'] == 'back':
        i = max(0, i - 1)
    elif result['label'] == 'next':
        i += 1
    elif result['label'] is not None:
        key = label_key(zone, date, idx)
        labels[key] = {
            'label': result['label'],
            'zone': zone,
            'date': date,
            'track_idx': int(idx),
        }
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"{zone} | {date} | track {idx} → {result['label']}  ({len(labels)} labeled)")
        if len(labels) % 50 == 0:
            threading.Thread(target=_retrain_in_background, daemon=True).start()
            print(f"[retrain] Triggered in background ({len(labels)} labels)")
        i += 1

if ds is not None:
    ds.close()

print(f"Done. {len(labels)} paths labeled, saved to {OUTPUT_PATH}")
