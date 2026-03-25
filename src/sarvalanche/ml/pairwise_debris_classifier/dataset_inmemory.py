"""In-memory training dataset for pairwise debris detector.

Holds pre-denoised VV/VH diffs + static stack in memory, builds
(N_INPUT, 128, 128) patches on-the-fly in __getitem__.

Assumes SAR imagery was already TV-denoised per-timestep via
preprocess_rtc before pair diffs were computed.

Requires multiprocessing.set_start_method('fork') for workers on macOS.
"""

import logging
import time as _time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sarvalanche.ml.pairwise_debris_classifier.channels import N_SAR, N_STATIC, sign_log1p
from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import extract_all_pairs

PATCH_SIZE = 128

log = logging.getLogger(__name__)


class PairwiseDebrisDataset(Dataset):
    """In-memory dataset — all arrays shared via fork.

    Parameters
    ----------
    pair_diffs : list of (vv_diff, vh_diff, valid_mask) tuples
        Pre-denoised dB diffs per pair, each (H, W) float32.
        valid_mask is (H, W) bool — True where SAR coverage exists.
    pair_metas : list of dicts
        Per-pair metadata with 'track', 'span_days', 't_start', 't_end'.
    anf_per_track : dict
        {track_id: (H, W) normalized ANF array}.
    static_scene : ndarray
        (N_STATIC, H, W) static terrain stack.
    positions : list of tuples
        (date_idx, has_debris, y0, x0, pair_idx).
    date_configs : list of dicts
        Per-date config with 'pos_pair_indices' (set) and 'debris_mask' (H, W).
    augment : bool
        Enable horizontal flip + noise augmentation.
    anf_raw_per_track : dict, optional
        {track_id: (H, W) raw ANF} for masking labels in layover zones.
    max_label_anf : float
        ANF threshold above which debris labels are masked out.
    """

    def __init__(self, pair_diffs, pair_metas, anf_per_track, static_scene,
                 positions, date_configs, augment=True,
                 anf_raw_per_track=None, max_label_anf=5.0):
        self.pair_diffs = pair_diffs
        self.pair_metas = pair_metas
        self.anf_per_track = anf_per_track
        self.anf_raw_per_track = anf_raw_per_track or {}
        self.max_label_anf = max_label_anf
        self.static_scene = static_scene
        self.positions = positions
        self.date_configs = date_configs
        self.augment = augment
        self.patch_size = PATCH_SIZE

        self.labels = [int(p[1] and p[4] in date_configs[p[0]]['pos_pair_indices'])
                       for p in positions]  # p[4] = pair_idx

        # Precompute curriculum difficulty per sample so the sampler can
        # exclude hard positives in early epochs instead of mislabeling them.
        # Only computed for positive (labeled) samples; negatives get level 0.
        # Level 0: always included. Level 1: epoch >= 10. Level 2: epoch >= 20.
        self._curriculum_level = self._compute_curriculum_levels()

    def _compute_curriculum_levels(self):
        """Assign a difficulty level to each sample based on SAR signal strength.

        Level 0: strong signal (p90 > 1.0, contrast > 0.3) — include from epoch 0.
        Level 1: moderate signal (p90 > 0.5) — include from epoch 10.
        Level 2: weak signal (p98 > 0.3) — include from epoch 20.
        Level 3: no detectable signal — never include as positive.

        Negatives always get level 0.
        """
        levels = []
        ps = self.patch_size
        for idx, (date_idx, has_debris, y0, x0, pair_idx) in enumerate(self.positions):
            if not self.labels[idx]:
                levels.append(0)
                continue

            vv_diff, vh_diff, valid_mask = self.pair_diffs[pair_idx]
            cfg = self.date_configs[date_idx]
            label_px = cfg['debris_mask'][y0:y0 + ps, x0:x0 + ps]
            valid = valid_mask[y0:y0 + ps, x0:x0 + ps]
            debris_covered = (label_px > 0.5) & valid

            vv = vv_diff[y0:y0 + ps, x0:x0 + ps].astype(np.float32)
            change_vv = sign_log1p(vv)
            debris_change = change_vv[debris_covered]

            if len(debris_change) == 0:
                levels.append(3)
                continue

            p90 = float(np.percentile(debris_change, 90))
            p98 = float(np.percentile(debris_change, 98))
            mn = float(debris_change.mean())
            contrast = p90 - mn

            if p90 > 1.0 and contrast > 0.3:
                levels.append(0)
            elif p90 > 0.5:
                levels.append(1)
            elif p98 > 0.3:
                levels.append(2)
            else:
                levels.append(3)

        return levels

    def get_valid_indices(self, epoch):
        """Return indices of samples that should be used at this epoch.

        Positive samples are gated by curriculum level. Negatives always included.
        """
        if epoch < 10:
            max_level = 0
        elif epoch < 20:
            max_level = 1
        else:
            max_level = 2
        return [i for i, lvl in enumerate(self._curriculum_level) if lvl <= max_level]

    def __len__(self):
        return len(self.positions)

    def _build_patch(self, pair_idx, y0, x0):
        """Build (N_INPUT, patch_size, patch_size) and valid mask from pair diffs + static."""
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]
        vv_diff, vh_diff, valid_mask = self.pair_diffs[pair_idx]
        H, W = vv_diff.shape

        y0c, x0c = max(y0, 0), max(x0, 0)
        y1, x1 = min(y0 + ps, H), min(x0 + ps, W)

        vv = vv_diff[y0c:y1, x0c:x1].astype(np.float32)
        vh = vh_diff[y0c:y1, x0c:x1].astype(np.float32)

        change_vv = sign_log1p(vv)
        change_vh = sign_log1p(vh)
        cr = vh - vv
        change_cr = sign_log1p(cr)
        anf = self.anf_per_track[meta['track']][y0c:y1, x0c:x1]

        sar = np.stack([change_vv, change_vh, change_cr, anf], axis=0)
        static = self.static_scene[:, y0c:y1, x0c:x1]
        valid = valid_mask[y0c:y1, x0c:x1]

        patch = np.concatenate([sar, static], axis=0)

        ah, aw = y1 - y0c, x1 - x0c
        if ah < ps or aw < ps:
            padded = np.zeros((N_SAR + N_STATIC, ps, ps), dtype=np.float32)
            valid_padded = np.zeros((ps, ps), dtype=bool)
            py, px = y0c - y0, x0c - x0
            padded[:, py:py + ah, px:px + aw] = patch
            valid_padded[py:py + ah, px:px + aw] = valid
            return padded, valid_padded

        return patch, valid

    def __getitem__(self, idx):
        date_idx, has_debris, y0, x0, pair_idx = self.positions[idx]
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]

        patch, valid = self._build_patch(pair_idx, y0, x0)

        cfg = self.date_configs[date_idx]
        is_bracketing = pair_idx in cfg['pos_pair_indices']

        if has_debris and is_bracketing:
            label_mask = cfg['debris_mask'][y0:y0 + ps, x0:x0 + ps].copy()
        else:
            label_mask = np.zeros((ps, ps), dtype=np.float32)

        # Zero out labels where there's no SAR coverage
        label_mask[~valid] = 0.0

        # Zero out labels in high-ANF layover/foreshortening zones
        if self.anf_raw_per_track and meta['track'] in self.anf_raw_per_track:
            anf_raw = self.anf_raw_per_track[meta['track']][y0:y0 + ps, x0:x0 + ps]
            label_mask[anf_raw >= self.max_label_anf] = 0.0

        if self.augment:
            if np.random.random() > 0.5:
                patch = patch[:, :, ::-1].copy()
                label_mask = label_mask[:, ::-1].copy()

        return {
            'x': torch.from_numpy(np.ascontiguousarray(patch)),
            'label': torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis])),
        }


def build_inmemory_dataset(
    ds, static_scene,
    date_polygon_pairs, geotiff_dirs=None,
    max_span_days=60,
    stride=64, neg_ratio=1.0, min_debris_frac=0.005, augment=True,
):
    """Build in-memory training dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Season dataset (pre-denoised via preprocess_rtc).
    static_scene : ndarray
        (N_STATIC, H, W) from build_static_stack.
    date_polygon_pairs : list of (date_str, GeoDataFrame, geotiff_dir)
        Per-date debris polygon labels.
    max_span_days : int
        Maximum pair span in days.
    stride : int
        Patch sampling stride.
    neg_ratio : float
        Ratio of negative to positive spatial positions per date.
        Each position is crossed with all pairs, so the final sample
        ratio matches this ratio.
    min_debris_frac : float
        Minimum debris fraction in a patch to count as positive.
    augment : bool
        Enable augmentation.
    """
    import rasterio
    import rasterio.features
    from rasterio.transform import from_bounds

    H, W = ds.sizes['y'], ds.sizes['x']
    x_arr, y_arr = ds.x.values, ds.y.values
    dx = abs(float(x_arr[1] - x_arr[0]))
    dy = abs(float(y_arr[1] - y_arr[0]))
    transform = from_bounds(
        float(x_arr.min()) - dx / 2, float(y_arr.min()) - dy / 2,
        float(x_arr.max()) + dx / 2, float(y_arr.max()) + dy / 2, W, H,
    )
    ps = PATCH_SIZE

    # Extract pairs using shared pair_extraction module
    log.info("Extracting pair diffs...")
    t0 = _time.time()
    pair_diffs, pair_metas, anf_per_track, anf_raw_per_track = extract_all_pairs(
        ds, max_span_days=max_span_days)
    log.info("  %d pairs in %.0fs (%.1f GB)", len(pair_diffs), _time.time() - t0,
             sum(vv.nbytes + vh.nbytes + m.nbytes for vv, vh, m in pair_diffs) / 1e9)

    # Per-date setup
    date_configs = []
    all_positions = []
    N = len(pair_metas)

    for di, (date_str, gdf, gt_dir) in enumerate(date_polygon_pairs):
        if gdf.crs and ds.rio.crs and gdf.crs != ds.rio.crs:
            gdf = gdf.to_crs(ds.rio.crs)
        if len(gdf) == 0:
            debris_mask = np.zeros((H, W), dtype=np.float32)
        else:
            debris_mask = rasterio.features.geometry_mask(
                gdf.geometry, out_shape=(H, W), transform=transform,
                invert=True, all_touched=True,
            ).astype(np.float32)

        # Determine reviewed area from geotiff footprints
        reviewed = np.ones((H, W), dtype=bool)
        if gt_dir is not None and gt_dir.is_dir():
            from shapely.geometry import box
            reviewed = np.zeros((H, W), dtype=bool)
            for tif in gt_dir.glob("*.tif"):
                with rasterio.open(tif) as src:
                    b = src.bounds
                reviewed |= rasterio.features.geometry_mask(
                    [box(b.left, b.bottom, b.right, b.top)],
                    out_shape=(H, W), transform=transform, invert=True,
                )

        ref = pd.Timestamp(date_str)
        pos_pair_indices = set()

        # Check if labels have pair constraints (autolabels with track/time info)
        has_pair_constraint = (
            len(gdf) > 0
            and 'track' in gdf.columns
            and 't_start' in gdf.columns
            and 't_end' in gdf.columns
        )

        if has_pair_constraint:
            allowed_pairs = set()
            for _, row in gdf[['track', 't_start', 't_end']].drop_duplicates().iterrows():
                label_track = str(int(row['track']))
                label_ts = pd.Timestamp(row['t_start'])
                label_te = pd.Timestamp(row['t_end'])
                for pi, meta in enumerate(pair_metas):
                    if (meta['track'] == label_track
                            and abs((meta['t_start'] - label_ts).days) <= 1
                            and abs((meta['t_end'] - label_te).days) <= 1):
                        allowed_pairs.add(pi)
            pos_pair_indices = allowed_pairs
            log.info("  %s: autolabel — %d constrained pairs", date_str, len(pos_pair_indices))
        else:
            for pi, meta in enumerate(pair_metas):
                if meta['t_start'] <= ref < meta['t_end']:
                    pos_pair_indices.add(pi)

        cfg = {'pos_pair_indices': pos_pair_indices, 'debris_mask': debris_mask}
        date_configs.append(cfg)

        # Sample patch positions
        debris_pos = []
        nondebris_pos = []
        for y0 in range(0, H - ps + 1, stride):
            for x0 in range(0, W - ps + 1, stride):
                if reviewed[y0:y0 + ps, x0:x0 + ps].mean() < 0.5:
                    continue
                frac = debris_mask[y0:y0 + ps, x0:x0 + ps].mean()
                if frac >= min_debris_frac:
                    debris_pos.append((y0, x0))
                else:
                    nondebris_pos.append((y0, x0))

        n_neg = min(int(len(debris_pos) * neg_ratio), len(nondebris_pos))
        if len(nondebris_pos) > n_neg:
            rng = np.random.default_rng(42 + di)
            idx = rng.choice(len(nondebris_pos), size=n_neg, replace=False)
            nondebris_pos = [nondebris_pos[i] for i in sorted(idx)]

        log.info("  %s: %d pos + %d neg positions, %d bracketing pairs",
                 date_str, len(debris_pos), len(nondebris_pos), len(pos_pair_indices))

        for y0, x0 in debris_pos:
            for pi in range(N):
                all_positions.append((di, True, y0, x0, pi))
        for y0, x0 in nondebris_pos:
            for pi in range(N):
                all_positions.append((di, False, y0, x0, pi))

    log.info("Total positions x pairs: %d", len(all_positions))

    return PairwiseDebrisDataset(
        pair_diffs, pair_metas, anf_per_track, static_scene, all_positions,
        date_configs, augment=augment, anf_raw_per_track=anf_raw_per_track,
    )


