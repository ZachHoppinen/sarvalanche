"""Training dataset for pairwise debris detector.

Uses memmapped .npy files (created by prepare_netcdf.py) for fast random
patch access. Falls back to NetCDF if .npy files don't exist.

Pair diffs are computed per patch (subtraction of two 128x128 slices from
memmapped arrays — microseconds, not milliseconds).

Requires:
  - NetCDFs prepared via prepare_netcdf.py (adds derived vars + exports .npy)
  - SAR imagery pre-denoised via preprocess_rtc
  - multiprocessing.set_start_method('fork') for workers on macOS

Augmentation: horizontal flip only.
"""

import logging
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from sarvalanche.ml.pairwise_debris_classifier.channels import (
    N_SAR, N_STATIC, STATIC_CHANNELS, normalize_anf,
    normalize_static_channel, sign_log1p,
)

PATCH_SIZE = 128

log = logging.getLogger(__name__)


def _parse_pair_from_filename(filename):
    """Extract (track, t_start, t_end) from a pair-labeled gpkg filename.

    Matches patterns like:
        avalanche_labels_trk129_2024-01-31_2024-02-12_11d.gpkg
        autolabels_2024-12-30.gpkg (no pair info)

    Returns (track_str, t_start_str, t_end_str) or None if no match.
    """
    import re
    m = re.search(r'trk(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_\d+d', str(filename))
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None

_NC_VAR_FOR_CHANNEL = {
    'slope': 'slope',
    'aspect_northing': 'aspect_northing',
    'aspect_easting': 'aspect_easting',
    'cell_counts': 'cell_counts',
    'tpi': 'tpi',
}


class PairwiseDebrisDataset(Dataset):
    """Dataset for pairwise debris detection training.

    Uses memmapped .npy files for VV/VH if available (fast), with
    pre-loaded static channels and ANF. Falls back to NetCDF reads
    if .npy files don't exist.

    Positions are 6-tuples: (date_idx, has_debris, y0, x0, pair_idx, is_auto).
    The is_auto flag allows the sampler to phase out autolabels.
    """

    def __init__(self, nc_path, pair_metas, positions, date_configs,
                 anf_track_indices=None, augment=True, max_label_anf=3.0,
                 sar_only=False, post_context=False):
        self.nc_path = Path(nc_path)
        self.pair_metas = pair_metas
        self.positions = positions
        self.date_configs = date_configs
        self.anf_track_indices = anf_track_indices or {}
        self.augment = augment
        self.max_label_anf = max_label_anf
        self.sar_only = sar_only
        self.post_context = post_context
        self.patch_size = PATCH_SIZE

        # Try to open memmapped .npy files for fast access
        vv_npy = self.nc_path.parent / (self.nc_path.stem + '_VV.npy')
        vh_npy = self.nc_path.parent / (self.nc_path.stem + '_VH.npy')
        static_npy = self.nc_path.parent / (self.nc_path.stem + '_static.npy')
        anf_npz = self.nc_path.parent / (self.nc_path.stem + '_anf.npy.npz')

        if vv_npy.exists():
            self._vv = np.load(vv_npy, mmap_mode='r')
            self._vh = np.load(vh_npy, mmap_mode='r') if vh_npy.exists() else None
            self._use_memmap = True
            log.info("  Using memmapped VV/VH from %s", vv_npy.parent.name)
        else:
            self._vv = None
            self._vh = None
            self._use_memmap = False
            log.info("  No .npy files found, using NetCDF (slow)")

        # Preload static channels (small — ~50MB per zone)
        if static_npy.exists():
            self._static = np.load(static_npy)  # (N_STATIC, H, W), pre-normalized
            log.info("  Static channels preloaded: %s", self._static.shape)
        else:
            self._static = None

        # Preload ANF per track (small)
        self._anf_norm = {}
        self._anf_raw = {}
        if anf_npz.exists():
            anf_data = np.load(anf_npz)
            anf_arrays = anf_data['data']   # (N_tracks, H, W)
            anf_tracks = anf_data['tracks']  # track IDs
            for si, track_id in enumerate(anf_tracks):
                track_str = str(int(track_id))
                raw = np.nan_to_num(anf_arrays[si].astype(np.float32), nan=1.0)
                self._anf_raw[track_str] = raw
                self._anf_norm[track_str] = normalize_anf(raw)
            log.info("  ANF preloaded: %d tracks", len(anf_tracks))

        # NetCDF fallback handle (opened lazily)
        self._ds = None

        # Precompute labels for sampler
        self.labels = [int(p[1] and p[4] in date_configs[p[0]]['pos_pair_indices'])
                       for p in positions]

        # Curriculum (computed lazily)
        self._curriculum_scores = None

    @property
    def ds(self):
        if self._ds is None:
            self._ds = xr.open_dataset(self.nc_path)
        return self._ds

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __del__(self):
        self.close()

    def _get_pair_diff(self, pair_idx, y0, x0, ps):
        """Compute VV/VH diffs and valid mask for a pair at a patch."""
        meta = self.pair_metas[pair_idx]
        ti = int(meta['ti_global'])
        tj = int(meta['tj_global'])

        if self._use_memmap:
            vv_i = self._vv[ti, y0:y0+ps, x0:x0+ps].astype(np.float32)
            vv_j = self._vv[tj, y0:y0+ps, x0:x0+ps].astype(np.float32)
        else:
            vv_i = (self.ds['VV']
                    .isel(time=ti, y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                    .values.astype(np.float32))
            vv_j = (self.ds['VV']
                    .isel(time=tj, y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                    .values.astype(np.float32))

        vv_diff_raw = vv_j - vv_i
        valid = np.isfinite(vv_diff_raw)
        vv_diff = np.nan_to_num(vv_diff_raw, nan=0.0)

        if self._use_memmap:
            if self._vh is not None:
                vh_i = self._vh[ti, y0:y0+ps, x0:x0+ps].astype(np.float32)
                vh_j = self._vh[tj, y0:y0+ps, x0:x0+ps].astype(np.float32)
                vh_diff_raw = vh_j - vh_i
                valid = valid & np.isfinite(vh_diff_raw)
                vh_diff = np.nan_to_num(vh_diff_raw, nan=0.0)
            else:
                vh_diff = np.zeros_like(vv_diff)
        else:
            if 'VH' in self.ds:
                vh_i = (self.ds['VH']
                        .isel(time=ti, y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                        .values.astype(np.float32))
                vh_j = (self.ds['VH']
                        .isel(time=tj, y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                        .values.astype(np.float32))
                vh_diff_raw = vh_j - vh_i
                valid = valid & np.isfinite(vh_diff_raw)
                vh_diff = np.nan_to_num(vh_diff_raw, nan=0.0)
            else:
                vh_diff = np.zeros_like(vv_diff)

        return vv_diff, vh_diff, valid

    def _get_anf_patch(self, track_str, y0, x0, ps):
        """Get normalized ANF and raw ANF for a patch."""
        if track_str in self._anf_norm:
            return (self._anf_norm[track_str][y0:y0+ps, x0:x0+ps],
                    self._anf_raw[track_str][y0:y0+ps, x0:x0+ps])
        if track_str in self.anf_track_indices:
            st_idx = self.anf_track_indices[track_str]
            anf_raw = (self.ds['anf']
                       .isel(static_track=st_idx, y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                       .values.astype(np.float32))
            anf_arr = np.nan_to_num(anf_raw, nan=1.0)
            return normalize_anf(anf_arr), anf_raw
        return np.ones((ps, ps), dtype=np.float32), None

    def _get_static_patch(self, y0, x0, ps):
        """Get normalized static channels for a patch."""
        if self._static is not None:
            return self._static[:, y0:y0+ps, x0:x0+ps].copy()
        static = np.zeros((N_STATIC, ps, ps), dtype=np.float32)
        for ch, var in enumerate(STATIC_CHANNELS):
            nc_var = _NC_VAR_FOR_CHANNEL.get(var)
            if nc_var and nc_var in self.ds:
                arr = (self.ds[nc_var]
                       .isel(y=slice(y0, y0+ps), x=slice(x0, x0+ps))
                       .values.astype(np.float32))
                arr = np.nan_to_num(arr, nan=0.0)
                static[ch] = normalize_static_channel(arr, var)
        return static

    def _get_post_patch(self, pair_idx, y0, x0, ps):
        """Get normalized post-event VV/VH for a patch. dB → [0, 1]."""
        meta = self.pair_metas[pair_idx]
        tj = int(meta['tj_global'])
        DB_MIN, DB_MAX = -30.0, 5.0

        if self._use_memmap:
            post_vv = self._vv[tj, y0:y0+ps, x0:x0+ps].astype(np.float32)
            post_vh = (self._vh[tj, y0:y0+ps, x0:x0+ps].astype(np.float32)
                       if self._vh is not None
                       else np.zeros((ps, ps), dtype=np.float32))
        else:
            ds = self.ds
            post_vv = ds['VV'].isel(time=tj, y=slice(y0, y0+ps),
                                    x=slice(x0, x0+ps)).values.astype(np.float32)
            post_vh = (ds['VH'].isel(time=tj, y=slice(y0, y0+ps),
                                     x=slice(x0, x0+ps)).values.astype(np.float32)
                       if 'VH' in ds else np.zeros((ps, ps), dtype=np.float32))

        post_vv = np.clip((np.nan_to_num(post_vv, nan=DB_MIN) - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
        post_vh = np.clip((np.nan_to_num(post_vh, nan=DB_MIN) - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
        return np.stack([post_vv, post_vh], axis=0)

    def _build_patch(self, pair_idx, y0, x0):
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]

        vv_diff, vh_diff, valid = self._get_pair_diff(pair_idx, y0, x0, ps)

        change_vv = sign_log1p(vv_diff)
        change_vh = sign_log1p(vh_diff)
        change_cr = sign_log1p(vh_diff - vv_diff)
        anf_norm, _ = self._get_anf_patch(meta['track'], y0, x0, ps)

        sar = np.stack([change_vv, change_vh, change_cr, anf_norm], axis=0)

        if self.post_context:
            post = self._get_post_patch(pair_idx, y0, x0, ps)
            sar = np.concatenate([sar, post], axis=0)

        if self.sar_only:
            return sar, valid

        static = self._get_static_patch(y0, x0, ps)
        patch = np.concatenate([sar, static], axis=0)
        return patch, valid

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        date_idx, has_debris, y0, x0, pair_idx, is_auto = self.positions[idx]
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]

        patch, valid = self._build_patch(pair_idx, y0, x0)

        cfg = self.date_configs[date_idx]
        is_bracketing = pair_idx in cfg['pos_pair_indices']

        if has_debris and is_bracketing:
            label_mask = cfg['debris_mask'][y0:y0 + ps, x0:x0 + ps].copy()
        else:
            label_mask = np.zeros((ps, ps), dtype=np.float32)

        label_mask[~valid] = 0.0

        _, anf_raw = self._get_anf_patch(meta['track'], y0, x0, ps)
        if anf_raw is not None:
            label_mask[anf_raw >= self.max_label_anf] = 0.0

        if self.augment:
            # Horizontal flip — negate aspect_easting
            if np.random.random() > 0.5:
                patch = patch[:, :, ::-1].copy()
                label_mask = label_mask[:, ::-1].copy()
                if not self.sar_only:
                    # aspect_easting is 3rd static channel (index 2 within static)
                    n_sar = 6 if self.post_context else 4
                    patch[n_sar + 2] = -patch[n_sar + 2]

        return {
            'x': torch.from_numpy(np.ascontiguousarray(patch)),
            'label': torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis])),
        }

    # ── Curriculum ───────────────────────────────────────────────────

    def get_valid_indices(self, epoch, total_epochs=100, auto_label_frac=0.6):
        """Return indices valid at this epoch.

        Curriculum: percentile-based. Epoch 0-9 uses top 20% of positives,
        10-19 uses top 50%, 20+ uses all (except no-signal).

        Auto-label phasing: after auto_label_frac * total_epochs, samples
        with is_auto=True are excluded.
        """
        if self._curriculum_scores is None:
            self._curriculum_scores = self._compute_curriculum_scores()

        # Percentile cutoff for positives
        if epoch < 10:
            pct_cutoff = 0.80  # top 20%
        elif epoch < 20:
            pct_cutoff = 0.50  # top 50%
        else:
            pct_cutoff = 0.0   # all

        # Auto-label phase cutoff
        auto_cutoff_epoch = int(total_epochs * auto_label_frac)
        exclude_auto = epoch >= auto_cutoff_epoch

        valid = []
        for i, pos in enumerate(self.positions):
            is_auto = pos[5]

            # Phase out autolabels after cutoff
            if exclude_auto and is_auto:
                continue

            if self.labels[i]:
                # Positive: check curriculum score percentile
                score = self._curriculum_scores[i]
                if score is None:
                    continue  # no signal — permanently excluded
                if score < pct_cutoff:
                    continue
            # Negatives always included (unless auto-excluded above)
            valid.append(i)

        return valid

    def _score_one_sample(self, idx):
        """Compute curriculum score for a single sample. Thread-safe."""
        if not self.labels[idx]:
            return 0.0

        ps = self.patch_size
        date_idx, has_debris, y0, x0, pair_idx, is_auto = self.positions[idx]
        meta = self.pair_metas[pair_idx]
        vv_diff, vh_diff, valid = self._get_pair_diff(pair_idx, y0, x0, ps)
        change_vv = sign_log1p(vv_diff)
        change_vh = sign_log1p(vh_diff)

        cfg = self.date_configs[date_idx]
        label_px = cfg['debris_mask'][y0:y0 + ps, x0:x0 + ps]
        debris_covered = (label_px > 0.5) & valid

        # Median ANF check — reject entire sample if layover
        _, anf_raw = self._get_anf_patch(meta['track'], y0, x0, ps)
        if anf_raw is not None:
            anf_in_debris = anf_raw[debris_covered]
            if len(anf_in_debris) > 0 and float(np.median(anf_in_debris)) > 3.0:
                return None
            debris_covered = debris_covered & (anf_raw < self.max_label_anf)

        vv_debris = change_vv[debris_covered]
        vh_debris = change_vh[debris_covered]

        if len(vv_debris) == 0:
            return None

        return max(float(np.percentile(vv_debris, 90)),
                   float(np.percentile(vh_debris, 90)))

    def _compute_curriculum_scores(self):
        """Compute signal scores for all samples, parallelized with threads.

        Threads work well here because numpy releases the GIL during array
        operations on memmapped data.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_pos = sum(self.labels)
        log.info("Computing curriculum scores for %d positive samples (threaded)...", n_pos)
        t0 = _time.time()

        n_workers = min(8, max(1, n_pos // 100))
        pos_indices = [i for i, l in enumerate(self.labels) if l]

        raw_scores = [0.0] * len(self.positions)  # default for negatives

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(self._score_one_sample, idx): idx
                       for idx in pos_indices}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                raw_scores[idx] = future.result()
                done += 1
                if done % 2000 == 0:
                    log.info("  Curriculum: %d/%d positive samples", done, n_pos)

        # Collect valid positive scores for percentile ranking
        pos_scores = [s for s in raw_scores if isinstance(s, float) and s > 0]

        if pos_scores:
            pos_scores_sorted = np.sort(pos_scores)
            percentile_scores = []
            for s in raw_scores:
                if s is None:
                    percentile_scores.append(None)
                elif isinstance(s, float) and s > 0:
                    rank = np.searchsorted(pos_scores_sorted, s) / len(pos_scores_sorted)
                    percentile_scores.append(rank)
                else:
                    percentile_scores.append(0.0)
        else:
            percentile_scores = raw_scores

        n_excluded = sum(1 for s in percentile_scores if s is None)
        # Count ANF-rejected vs no-signal by re-checking (cheap since scores are cached)
        n_anf_rejected = 0
        n_no_signal = 0
        for idx in pos_indices:
            if raw_scores[idx] is None:
                pos = self.positions[idx]
                meta = self.pair_metas[pos[4]]
                _, anf_raw = self._get_anf_patch(meta['track'], pos[2], pos[3], self.patch_size)
                if anf_raw is not None:
                    cfg = self.date_configs[pos[0]]
                    label_px = cfg['debris_mask'][pos[2]:pos[2]+self.patch_size, pos[3]:pos[3]+self.patch_size]
                    # Quick check: was it ANF or no-signal?
                    debris_px = label_px > 0.5
                    if debris_px.sum() > 0 and float(np.median(anf_raw[debris_px])) > 2.0:
                        n_anf_rejected += 1
                    else:
                        n_no_signal += 1
                else:
                    n_no_signal += 1
        log.info("  Curriculum scores: %d positives ranked, %d excluded "
                 "(%d ANF>3.0 median, %d no-signal) — %.0fs",
                 len(pos_scores), n_excluded, n_anf_rejected, n_no_signal,
                 _time.time() - t0)

        return percentile_scores


def build_lazy_dataset(
    nc_path, date_polygon_pairs,
    max_span_days=60, stride=64, neg_ratio=1.0, min_debris_frac=0.005,
    augment=True, month_range=(10, 4), sar_only=False, post_context=False,
):
    """Build a training dataset from a single season NetCDF.

    Negatives are only sampled from human-labeled dates (not autolabel dates)
    since autolabeled areas may contain unlabeled positives.

    Positions are 6-tuples: (date_idx, has_debris, y0, x0, pair_idx, is_auto).

    Parameters
    ----------
    date_polygon_pairs : list of tuples
        Each entry is (date_str, GeoDataFrame, geotiff_dir, is_auto) or
        (date_str, GeoDataFrame, geotiff_dir). If is_auto is omitted,
        it's inferred from pair constraint columns/filename.
    month_range : tuple of (start_month, end_month)
        Only include pairs where t_end falls within this month range.
        Wraps around year boundary. Default (10, 4) = October through April.
        Set to None to include all months.
    """
    import rasterio
    import rasterio.features
    from rasterio.transform import from_bounds

    nc_path = Path(nc_path)
    t_total = _time.time()
    log.info("  Opening %s for metadata", nc_path.name)
    ds = xr.open_dataset(nc_path)

    H, W = ds.sizes['y'], ds.sizes['x']
    log.info("  Scene: %d times, %dx%d", ds.sizes.get('time', 0), H, W)
    x_arr, y_arr = ds.x.values, ds.y.values
    dx = abs(float(x_arr[1] - x_arr[0]))
    dy = abs(float(y_arr[1] - y_arr[0]))
    transform = from_bounds(
        float(x_arr.min()) - dx / 2, float(y_arr.min()) - dy / 2,
        float(x_arr.max()) + dx / 2, float(y_arr.max()) + dy / 2, W, H,
    )
    ps = PATCH_SIZE

    # Build pair metadata
    log.info("  Building pair metadata (max span %dd)...", max_span_days)
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds['track'].values if 'track' in ds else np.zeros(len(times), dtype=int)

    track_ids = sorted(set(str(int(t)) for t in np.unique(tracks)))
    track_global_indices = {}
    for track_str in track_ids:
        track_global_indices[track_str] = np.where(tracks == int(track_str))[0]

    anf_track_indices = {}
    if 'anf' in ds:
        static_tracks = ds['anf'].static_track.values
        for track_str in track_ids:
            track_int = int(track_str)
            if track_int in static_tracks:
                anf_track_indices[track_str] = int(np.where(static_tracks == track_int)[0][0])

    # Month filter: determine which months are valid
    if month_range is not None:
        m_start, m_end = month_range
        if m_start <= m_end:
            valid_months = set(range(m_start, m_end + 1))
        else:
            # Wraps around year boundary (e.g. Oct-Apr = 10,11,12,1,2,3,4)
            valid_months = set(range(m_start, 13)) | set(range(1, m_end + 1))
        log.info("  Month filter: %s (months %s)", month_range,
                 ','.join(str(m) for m in sorted(valid_months)))
    else:
        valid_months = None

    pair_metas = []
    n_month_filtered = 0
    for track_str in track_ids:
        global_idx = track_global_indices[track_str]
        track_times = times[global_idx]
        for a in range(len(track_times)):
            for b in range(a + 1, len(track_times)):
                span = (track_times[b] - track_times[a]).days
                if span < 1 or span > max_span_days:
                    continue
                # Filter by month of pair end date
                if valid_months is not None and track_times[b].month not in valid_months:
                    n_month_filtered += 1
                    continue
                pair_metas.append({
                    'track': track_str,
                    'i': a, 'j': b,
                    'ti_global': int(global_idx[a]),
                    'tj_global': int(global_idx[b]),
                    't_start': track_times[a],
                    't_end': track_times[b],
                    'span_days': span,
                })

    for track_str in track_ids:
        n_track_pairs = sum(1 for m in pair_metas if m['track'] == track_str)
        n_times = len(track_global_indices[track_str])
        log.info("    Track %s: %d times, %d pairs", track_str, n_times, n_track_pairs)
    log.info("  Total: %d pairs across %d tracks", len(pair_metas), len(track_ids))
    if n_month_filtered > 0:
        log.info("  Filtered %d pairs outside month range %s", n_month_filtered, month_range)

    # Per-date setup
    date_configs = []
    all_positions = []
    N = len(pair_metas)
    max_neg_pairs_per_pos = min(N, 10)
    skipped_dates = 0

    for di, entry in enumerate(date_polygon_pairs):
        # Unpack — support both 3-tuple and 4-tuple
        if len(entry) == 4:
            date_str, gdf, gt_dir, is_auto = entry
        else:
            date_str, gdf, gt_dir = entry
            is_auto = None  # will be inferred below

        if gdf.crs and ds.rio.crs and str(gdf.crs) != str(ds.rio.crs):
            gdf = gdf.to_crs(ds.rio.crs)
        if len(gdf) == 0:
            debris_mask = np.zeros((H, W), dtype=np.float32)
        else:
            debris_mask = rasterio.features.geometry_mask(
                gdf.geometry, out_shape=(H, W), transform=transform,
                invert=True, all_touched=True,
            ).astype(np.float32)

        reviewed = np.ones((H, W), dtype=bool)
        if gt_dir is not None and Path(gt_dir).is_dir():
            from shapely.geometry import box
            reviewed = np.zeros((H, W), dtype=bool)
            for tif in Path(gt_dir).glob("*.tif"):
                with rasterio.open(tif) as src:
                    b = src.bounds
                reviewed |= rasterio.features.geometry_mask(
                    [box(b.left, b.bottom, b.right, b.top)],
                    out_shape=(H, W), transform=transform, invert=True,
                )

        ref = pd.Timestamp(date_str)
        pos_pair_indices = set()

        # Determine pair constraint: from gpkg columns, filename, or none
        pair_info_from_cols = (
            len(gdf) > 0
            and 'track' in gdf.columns
            and 't_start' in gdf.columns
            and 't_end' in gdf.columns
            and gdf['track'].notna().any()
        )

        # Try parsing pair info from the gpkg filename if available
        pair_info_from_filename = None
        if hasattr(gdf, 'attrs') and 'source' in gdf.attrs:
            pair_info_from_filename = _parse_pair_from_filename(gdf.attrs['source'])
        # Also try from date_str if it looks like a filename was used
        if pair_info_from_filename is None:
            pair_info_from_filename = _parse_pair_from_filename(date_str)

        has_pair_constraint = pair_info_from_cols or pair_info_from_filename is not None

        # Infer is_auto if not provided by caller
        if is_auto is None:
            is_auto = has_pair_constraint  # legacy behavior for autolabels

        if pair_info_from_cols:
            # Use gpkg columns (autolabel format)
            valid_rows = gdf[gdf['track'].notna()]
            for _, row in valid_rows[['track', 't_start', 't_end']].drop_duplicates().iterrows():
                label_track = str(int(float(row['track'])))
                label_ts = pd.Timestamp(row['t_start'])
                label_te = pd.Timestamp(row['t_end'])
                for pi, meta in enumerate(pair_metas):
                    if (meta['track'] == label_track
                            and abs((meta['t_start'] - label_ts).days) <= 1
                            and abs((meta['t_end'] - label_te).days) <= 1):
                        pos_pair_indices.add(pi)
        elif pair_info_from_filename is not None:
            # Use filename-parsed pair info
            label_track, label_ts_str, label_te_str = pair_info_from_filename
            label_ts = pd.Timestamp(label_ts_str)
            label_te = pd.Timestamp(label_te_str)
            for pi, meta in enumerate(pair_metas):
                if (meta['track'] == label_track
                        and abs((meta['t_start'] - label_ts).days) <= 1
                        and abs((meta['t_end'] - label_te).days) <= 1):
                    pos_pair_indices.add(pi)
        else:
            # Date-based bracketing (human labels without pair info)
            for pi, meta in enumerate(pair_metas):
                if meta['t_start'] <= ref < meta['t_end']:
                    pos_pair_indices.add(pi)

        if len(pos_pair_indices) == 0:
            log.warning("  %s: 0 bracketing pairs — skipping", date_str)
            skipped_dates += 1
            date_configs.append({'pos_pair_indices': set(), 'debris_mask': np.zeros((H, W), dtype=np.float32)})
            continue

        cfg = {'pos_pair_indices': pos_pair_indices, 'debris_mask': debris_mask}
        date_configs.append(cfg)

        # Sample positions
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

        if len(debris_pos) == 0:
            if is_auto:
                log.info("  %s: autolabel — 0 pos positions (polys too small), skipping", date_str)
            else:
                log.info("  %s: 0 pos positions, skipping", date_str)
            continue

        # Positives: cross with bracketing pairs
        for y0, x0 in debris_pos:
            for pi in pos_pair_indices:
                all_positions.append((di, True, y0, x0, pi, is_auto))

        # Negatives: ONLY from human-labeled dates (not autolabels)
        if not is_auto:
            n_neg = min(int(len(debris_pos) * neg_ratio), len(nondebris_pos))
            if len(nondebris_pos) > n_neg:
                rng = np.random.default_rng(42 + di)
                idx = rng.choice(len(nondebris_pos), size=n_neg, replace=False)
                nondebris_pos = [nondebris_pos[i] for i in sorted(idx)]

            neg_rng = np.random.default_rng(42 + di + 1000)
            neg_pair_pool = list(range(N))
            for y0, x0 in nondebris_pos:
                if N <= max_neg_pairs_per_pos:
                    neg_pairs = neg_pair_pool
                else:
                    neg_pairs = neg_rng.choice(neg_pair_pool, size=max_neg_pairs_per_pos, replace=False)
                for pi in neg_pairs:
                    all_positions.append((di, False, y0, x0, int(pi), False))

            n_neg_for_date = len(nondebris_pos) * min(N, max_neg_pairs_per_pos)
        else:
            n_neg_for_date = 0

        n_pos_for_date = len(debris_pos) * len(pos_pair_indices)
        label_type = "autolabel" if is_auto else "manual"
        log.info("  %s: %s — %d pos + %d neg samples, %d pairs",
                 date_str, label_type, n_pos_for_date, n_neg_for_date, len(pos_pair_indices))

    n_pos_samples = sum(1 for p in all_positions if p[1])
    n_neg_samples = len(all_positions) - n_pos_samples
    n_auto = sum(1 for p in all_positions if p[5])
    if skipped_dates > 0:
        log.warning("  Skipped %d dates with 0 bracketing pairs", skipped_dates)
    log.info("  Total: %d samples (%d pos, %d neg, %d auto-labeled) from %d dates — %.0fs",
             len(all_positions), n_pos_samples, n_neg_samples, n_auto,
             len(date_configs), _time.time() - t_total)

    ds.close()

    return PairwiseDebrisDataset(
        nc_path, pair_metas, all_positions, date_configs,
        anf_track_indices=anf_track_indices, augment=augment,
        sar_only=sar_only, post_context=post_context,
    )
