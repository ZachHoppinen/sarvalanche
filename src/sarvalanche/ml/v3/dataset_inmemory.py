"""In-memory v3 Dataset — computes patches on-the-fly from scene arrays.

No disk extraction needed. Holds track VV/VH diffs + static stack in memory,
computes per-pair SAR patches in __getitem__.

Requires multiprocessing.set_start_method('fork') for workers on macOS.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sarvalanche.ml.v3.channels import N_SAR, N_STATIC, STATIC_CHANNELS
from sarvalanche.ml.v3.patch_extraction import V3_PATCH_SIZE, normalize_dem_patch

log = logging.getLogger(__name__)

SLOPE_IDX = STATIC_CHANNELS.index('slope')
DEM_IDX = STATIC_CHANNELS.index('dem')
CELL_COUNTS_IDX = STATIC_CHANNELS.index('cell_counts')


class V3InMemoryDataset(Dataset):
    """In-memory dataset — all arrays shared via fork."""

    def __init__(self, pair_diffs, pair_metas, anf_per_track, static_scene,
                 positions, date_configs, augment=True, regional_cache=None,
                 anf_raw_per_track=None, max_label_anf=5.0):
        self.pair_diffs = pair_diffs  # list of (vv_diff, vh_diff) tuples
        self.pair_metas = pair_metas
        self.anf_per_track = anf_per_track
        self.anf_raw_per_track = anf_raw_per_track or {}
        self.max_label_anf = max_label_anf
        self.static_scene = static_scene
        self.positions = positions
        self.date_configs = date_configs
        self.augment = augment
        self.patch_size = V3_PATCH_SIZE
        self.epoch = 0  # curriculum learning
        self.regional_cache = regional_cache or {}  # {pair_idx: (C, 128, 128)}

        self.labels = [int(p[1] and p[4] in date_configs[p[0]]['pos_pair_indices'])
                       for p in positions]

    def __len__(self):
        return len(self.positions)

    def _make_sar_static(self, pair_idx, y0, x0, size):
        """Build SAR + static for a given window. Returns (N_SAR, size, size), (N_STATIC, size, size)."""
        meta = self.pair_metas[pair_idx]
        vv_diff, vh_diff, vv_smooth, vh_smooth = self.pair_diffs[pair_idx]
        H, W = vv_diff.shape

        # Clamp to scene
        y0c, x0c = max(y0, 0), max(x0, 0)
        y1, x1 = min(y0 + size, H), min(x0 + size, W)

        vv = vv_diff[y0c:y1, x0c:x1].astype(np.float32)
        vh = vh_diff[y0c:y1, x0c:x1].astype(np.float32)

        change_vv = np.sign(vv) * np.log1p(np.abs(vv))
        change_vh = np.sign(vh) * np.log1p(np.abs(vh))
        cr = vh - vv
        change_cr = np.sign(cr) * np.log1p(np.abs(cr))
        anf = self.anf_per_track[meta['track']][y0c:y1, x0c:x1]
        prox = np.full(vv.shape, 1.0 / (1.0 + meta['span_days'] / 12.0), dtype=np.float32)

        # Smoothed channels (TV-denoised, then log1p like raw)
        vvs = vv_smooth[y0c:y1, x0c:x1].astype(np.float32)
        vhs = vh_smooth[y0c:y1, x0c:x1].astype(np.float32)
        change_vv_s = np.sign(vvs) * np.log1p(np.abs(vvs))
        change_vh_s = np.sign(vhs) * np.log1p(np.abs(vhs))

        # Coverage mask: 1 where we have real SAR data, 0 where NaN-filled
        coverage = (np.abs(vv) > 1e-6).astype(np.float32)

        sar = np.stack([change_vv, change_vh, change_cr, anf, prox,
                        change_vv_s, change_vh_s, coverage], axis=0)
        static = self.static_scene[:, y0c:y1, x0c:x1].copy()
        static = normalize_dem_patch(static)

        # Pad if at scene edge
        ah, aw = y1 - y0c, x1 - x0c
        if ah < size or aw < size:
            sar_p = np.zeros((N_SAR, size, size), dtype=np.float32)
            sta_p = np.zeros((N_STATIC, size, size), dtype=np.float32)
            py, px = y0c - y0, x0c - x0
            sar_p[:, py:py+ah, px:px+aw] = sar
            sta_p[:, py:py+ah, px:px+aw] = static
            return sar_p, sta_p

        return sar, static

    def __getitem__(self, idx):
        date_idx, has_debris, y0, x0, pair_idx, conf = self.positions[idx]
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]

        # Fine patch: 128×128 at full res
        sar_patch, static_patch = self._make_sar_static(pair_idx, y0, x0, ps)

        # Local context: 512×512 centered, downsampled to 128×128
        ctx_size = ps * 4
        ctx_y0 = y0 - (ctx_size - ps) // 2
        ctx_x0 = x0 - (ctx_size - ps) // 2
        ctx_sar, ctx_static = self._make_sar_static(pair_idx, ctx_y0, ctx_x0, ctx_size)
        ctx_sar = ctx_sar.reshape(N_SAR, ps, 4, ps, 4).mean(axis=(2, 4))
        ctx_static = ctx_static.reshape(N_STATIC, ps, 4, ps, 4).mean(axis=(2, 4))

        # Regional: precomputed whole-scene downsample
        if pair_idx in self.regional_cache:
            regional = self.regional_cache[pair_idx]
        else:
            regional = np.zeros((N_SAR + N_STATIC, ps, ps), dtype=np.float32)

        # Label with curriculum
        cfg = self.date_configs[date_idx]
        is_bracketing = pair_idx in cfg['pos_pair_indices']

        if has_debris and is_bracketing:
            label_px = cfg['debris_mask'][y0:y0+ps, x0:x0+ps]
            has_coverage = np.abs(sar_patch[0]) > 1e-6
            debris_covered = (label_px > 0.5) & has_coverage
            debris_change = sar_patch[0][debris_covered]
            if len(debris_change) > 0:
                p90 = float(np.percentile(debris_change, 90))
                mn = float(debris_change.mean())
                contrast = p90 - mn
                if self.epoch < 10:
                    passes = p90 > 1.0 and contrast > 0.3
                elif self.epoch < 20:
                    passes = p90 > 0.5
                else:
                    passes = np.percentile(debris_change, 98) > 0.3
            else:
                passes = False
            label_mask = label_px.copy() if passes else np.zeros((ps, ps), dtype=np.float32)
        else:
            label_mask = np.zeros((ps, ps), dtype=np.float32)

        no_cov = np.abs(sar_patch[0]) < 1e-6
        label_mask[no_cov] = 0.0

        # Mask debris in high-ANF zones (layover/foreshortening) for this track
        if self.anf_raw_per_track and meta['track'] in self.anf_raw_per_track:
            anf_raw = self.anf_raw_per_track[meta['track']][y0:y0+ps, x0:x0+ps]
            label_mask[anf_raw >= self.max_label_anf] = 0.0

        # Augmentation (same flip for fine + context, NOT regional)
        if self.augment:
            if np.random.random() > 0.5:
                sar_patch = sar_patch[:, :, ::-1].copy()
                static_patch = static_patch[:, :, ::-1].copy()
                ctx_sar = ctx_sar[:, :, ::-1].copy()
                ctx_static = ctx_static[:, :, ::-1].copy()
                label_mask = label_mask[:, ::-1].copy()
            sar_patch[0] += np.random.randn(*sar_patch[0].shape).astype(np.float32) * 0.05

        fine = np.concatenate([sar_patch, static_patch], axis=0)
        local_ctx = np.concatenate([ctx_sar, ctx_static], axis=0)

        return {
            'x': torch.from_numpy(np.ascontiguousarray(fine)),
            'local_ctx': torch.from_numpy(np.ascontiguousarray(local_ctx)),
            'regional': torch.from_numpy(np.ascontiguousarray(regional)),
            'label': torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis])),
            'confidence': torch.tensor(conf, dtype=torch.float32),
        }


def precompute_pair_diffs(tracks, pair_metas, tv_weight=1.0):
    """Precompute raw VV/VH diffs and TV-smoothed versions for all pairs."""
    from skimage.restoration import denoise_tv_chambolle

    pair_diffs = []
    for pi, meta in enumerate(pair_metas):
        td = tracks[meta['track']]
        vv_diff = np.nan_to_num(
            (td['vv'][meta['j']] - td['vv'][meta['i']]).astype(np.float32), nan=0.0)
        vh_diff = np.nan_to_num(
            (td['vh'][meta['j']] - td['vh'][meta['i']]).astype(np.float32), nan=0.0) if td['vh'] is not None else np.zeros_like(vv_diff)

        # TV-denoise the raw dB diffs
        vv_smooth = denoise_tv_chambolle(vv_diff, weight=tv_weight).astype(np.float32)
        vh_smooth = denoise_tv_chambolle(vh_diff, weight=tv_weight).astype(np.float32)

        pair_diffs.append((vv_diff, vh_diff, vv_smooth, vh_smooth))
        if (pi + 1) % 100 == 0:
            log.info("    Precomputed %d/%d pair diffs", pi + 1, len(pair_metas))
    return pair_diffs


def build_inmemory_dataset(
    ds, pair_metas, tracks, hrrr_cache, static_scene,
    date_polygon_pairs, val_path_mask, geotiff_dirs=None,
    stride=64, neg_ratio=1.0, min_debris_frac=0.005, augment=True,
):
    """Build in-memory dataset."""
    import geopandas as gpd
    import rasterio
    import rasterio.features
    from rasterio.transform import from_bounds
    from shapely.geometry import box
    import time as _time

    H, W = ds.sizes['y'], ds.sizes['x']
    x_arr, y_arr = ds.x.values, ds.y.values
    dx, dy = abs(float(x_arr[1] - x_arr[0])), abs(float(y_arr[1] - y_arr[0]))
    transform = from_bounds(
        float(x_arr.min()) - dx/2, float(y_arr.min()) - dy/2,
        float(x_arr.max()) + dx/2, float(y_arr.max()) + dy/2, W, H,
    )
    ps = V3_PATCH_SIZE

    # Precompute pair diffs
    log.info("Precomputing pair diffs...")
    t0 = _time.time()
    pair_diffs = precompute_pair_diffs(tracks, pair_metas)
    log.info("  Done in %.0fs (%.1f GB)", _time.time() - t0,
             sum(sum(a.nbytes for a in d) for d in pair_diffs) / 1e9)

    anf_per_track = {tid: td['anf'] for tid, td in tracks.items()}

    # Raw ANF for label masking (exclude high-ANF layover zones)
    anf_raw_per_track = {}
    if 'anf' in ds:
        static_tracks = ds.static_track.values
        anf_data = ds['anf'].values
        for si, st in enumerate(static_tracks):
            anf_raw_per_track[str(int(st))] = anf_data[si].squeeze().astype(np.float32)

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

        reviewed = np.ones((H, W), dtype=bool)
        if gt_dir is not None and gt_dir.is_dir():
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

        # Check if this GeoPackage has pair constraints (autolabels)
        # If so, only allow pairs matching the track + time window that generated the label
        has_pair_constraint = (
            len(gdf) > 0
            and 'track' in gdf.columns
            and 't_start' in gdf.columns
            and 't_end' in gdf.columns
        )

        if has_pair_constraint:
            # Autolabel: only match pairs from the same track/time windows
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
            # Human label: all bracketing pairs
            for pi, meta in enumerate(pair_metas):
                if meta['t_start'] <= ref < meta['t_end']:
                    pos_pair_indices.add(pi)

        cfg = {'pos_pair_indices': pos_pair_indices, 'debris_mask': debris_mask}
        date_configs.append(cfg)

        debris_pos = []
        nondebris_pos = []
        for y0 in range(0, H - ps + 1, stride):
            for x0 in range(0, W - ps + 1, stride):
                if reviewed[y0:y0+ps, x0:x0+ps].mean() < 0.5:
                    continue
                frac = debris_mask[y0:y0+ps, x0:x0+ps].mean()
                if frac >= min_debris_frac:
                    conf = _compute_confidence(static_scene[:, y0:y0+ps, x0:x0+ps],
                                                debris_mask[y0:y0+ps, x0:x0+ps], date_str)
                    debris_pos.append((y0, x0, conf))
                else:
                    nondebris_pos.append((y0, x0))

        n_neg = min(int(len(debris_pos) * neg_ratio), len(nondebris_pos))
        if len(nondebris_pos) > n_neg:
            rng = np.random.default_rng(42 + di)
            idx = rng.choice(len(nondebris_pos), size=n_neg, replace=False)
            nondebris_pos = [nondebris_pos[i] for i in sorted(idx)]

        log.info("  %s: %d pos + %d neg positions, %d bracketing pairs",
                 date_str, len(debris_pos), len(nondebris_pos), len(pos_pair_indices))

        for y0, x0, conf in debris_pos:
            for pi in range(N):
                all_positions.append((di, True, y0, x0, pi, conf))
        for y0, x0 in nondebris_pos:
            for pi in range(N):
                all_positions.append((di, False, y0, x0, pi, 1.0))

    log.info("Total positions × pairs: %d", len(all_positions))

    # Precompute regional views: whole scene downsampled to 128×128 per pair
    log.info("Precomputing regional views...")
    t0 = _time.time()
    target = V3_PATCH_SIZE
    from skimage.transform import resize
    static_regional = resize(static_scene.transpose(1, 2, 0),
                             (target, target), order=1,
                             preserve_range=True).transpose(2, 0, 1).astype(np.float32)
    regional_cache = {}
    for pi, (vv_diff, vh_diff, vv_sm, vh_sm) in enumerate(pair_diffs):
        meta = pair_metas[pi]
        vv_s = resize(vv_diff, (target, target), order=1, preserve_range=True).astype(np.float32)
        vh_s = resize(vh_diff, (target, target), order=1, preserve_range=True).astype(np.float32)
        cv = np.sign(vv_s) * np.log1p(np.abs(vv_s))
        ch = np.sign(vh_s) * np.log1p(np.abs(vh_s))
        cr = vh_s - vv_s
        ccr = np.sign(cr) * np.log1p(np.abs(cr))
        anf_s = resize(anf_per_track[meta['track']], (target, target),
                        order=1, preserve_range=True).astype(np.float32)
        prox = np.full((target, target), 1.0 / (1.0 + meta['span_days'] / 12.0), dtype=np.float32)
        # Smoothed channels for regional
        vv_sm_s = resize(vv_sm, (target, target), order=1, preserve_range=True).astype(np.float32)
        vh_sm_s = resize(vh_sm, (target, target), order=1, preserve_range=True).astype(np.float32)
        cvs = np.sign(vv_sm_s) * np.log1p(np.abs(vv_sm_s))
        chs = np.sign(vh_sm_s) * np.log1p(np.abs(vh_sm_s))
        # Coverage: fraction of valid pixels at regional scale
        cov = (np.abs(vv_diff) > 1e-6).astype(np.float32)
        cov_r = resize(cov, (target, target), order=1, preserve_range=True).astype(np.float32)
        sar_r = np.stack([cv, ch, ccr, anf_s, prox, cvs, chs, cov_r], axis=0)
        regional_cache[pi] = np.concatenate([sar_r, static_regional], axis=0)
    log.info("  Regional: %d pairs (%.0fs, %.0f MB)", len(regional_cache),
             _time.time() - t0, sum(v.nbytes for v in regional_cache.values()) / 1e6)

    return V3InMemoryDataset(
        pair_diffs, pair_metas, anf_per_track, static_scene, all_positions,
        date_configs, augment=augment, regional_cache=regional_cache,
        anf_raw_per_track=anf_raw_per_track,
    )


def _compute_confidence(static_patch, label_patch, date_str):
    debris_px = label_patch > 0.5
    n_debris = int(debris_px.sum())
    if n_debris < 3:
        return 1.0
    slope = static_patch[SLOPE_IDX]
    valid = np.isfinite(slope) & (slope > 0.01)
    n_valid = int(valid.sum())
    if n_valid < 10:
        return 1.0
    coverage = n_debris / max(n_valid, 1)
    cov_conf = 0.1 if coverage > 0.40 else (0.3 if coverage > 0.25 else (0.6 if coverage > 0.15 else 1.0))
    dem = static_patch[DEM_IDX]
    dem_debris = dem[debris_px]
    dem_valid = dem[valid]
    if np.isfinite(dem_debris).sum() > 2 and np.isfinite(dem_valid).sum() > 10:
        dem_offset = float(np.nanmean(dem_debris)) - float(np.nanmean(dem_valid))
        elev_conf = 0.5 if dem_offset > 0.15 else (0.7 if dem_offset > 0.05 else 1.0)
    else:
        elev_conf = 1.0
    cc = static_patch[CELL_COUNTS_IDX]
    cc_debris = cc[debris_px]
    if np.isfinite(cc_debris).sum() > 0:
        rf = float((cc_debris > 0).sum()) / max(n_debris, 1)
        runout_conf = 1.0 if rf > 0.3 else (0.8 if rf > 0.1 else 0.5)
    else:
        runout_conf = 0.7
    month = int(date_str[5:7])
    spring_conf = 0.3 if month >= 4 else (0.7 if month == 3 else 1.0)
    return float(min(cov_conf, elev_conf, runout_conf, spring_conf))
