"""Siamese debris detection dataset.

Returns pre-event and post-event SAR patches separately instead of
pre-computed diffs. The model learns what features to diff.

Reuses the same lazy memmap infrastructure, pair metadata, position
sampling, curriculum scoring, and label handling from the pairwise
classifier. Only __getitem__ and _build_patch differ.
"""

import logging

import numpy as np
import torch

from sarvalanche.ml.pairwise_debris_classifier.dataset import (
    PATCH_SIZE,
    PairwiseDebrisDataset,
    build_lazy_dataset as _build_lazy_dataset_base,
)
from sarvalanche.ml.siamese_debris_classifier.channels import normalize_db

log = logging.getLogger(__name__)


class SiameseDebrisDataset(PairwiseDebrisDataset):
    """Dataset that returns (pre, post, static, label) instead of (concat, label).

    Inherits all infrastructure from PairwiseDebrisDataset:
    - Memmap VV/VH loading
    - Static channel preloading
    - ANF handling
    - Curriculum scoring
    - Position and pair metadata
    """

    def _build_branch_patch(self, time_idx, track_str, y0, x0):
        """Build a single branch patch: (VV_dB, VH_dB, ANF_norm) normalized."""
        ps = self.patch_size

        if self._use_memmap:
            vv = self._vv[time_idx, y0:y0 + ps, x0:x0 + ps].astype(np.float32)
            vh = (self._vh[time_idx, y0:y0 + ps, x0:x0 + ps].astype(np.float32)
                  if self._vh is not None
                  else np.zeros((ps, ps), dtype=np.float32))
        else:
            ds = self.ds
            vv = ds['VV'].isel(time=time_idx, y=slice(y0, y0 + ps),
                               x=slice(x0, x0 + ps)).values.astype(np.float32)
            vh = (ds['VH'].isel(time=time_idx, y=slice(y0, y0 + ps),
                                x=slice(x0, x0 + ps)).values.astype(np.float32)
                  if 'VH' in ds else np.zeros((ps, ps), dtype=np.float32))

        valid = np.isfinite(vv)
        vv = normalize_db(np.nan_to_num(vv, nan=-30.0))
        vh = normalize_db(np.nan_to_num(vh, nan=-30.0))

        anf_norm, _ = self._get_anf_patch(track_str, y0, x0, ps)

        return np.stack([vv, vh, anf_norm], axis=0), valid

    def __getitem__(self, idx):
        date_idx, has_debris, y0, x0, pair_idx, is_auto = self.positions[idx]
        ps = self.patch_size
        meta = self.pair_metas[pair_idx]

        ti = int(meta['ti_global'])
        tj = int(meta['tj_global'])
        track_str = meta['track']

        # Build pre and post branch patches
        pre_patch, valid_pre = self._build_branch_patch(ti, track_str, y0, x0)
        post_patch, valid_post = self._build_branch_patch(tj, track_str, y0, x0)
        valid = valid_pre & valid_post

        # Static channels (shared)
        if not self.sar_only:
            static = self._get_static_patch(y0, x0, ps)
        else:
            static = None

        # Label
        cfg = self.date_configs[date_idx]
        is_bracketing = pair_idx in cfg['pos_pair_indices']

        if has_debris and is_bracketing:
            label_mask = cfg['debris_mask'][y0:y0 + ps, x0:x0 + ps].copy()
        else:
            label_mask = np.zeros((ps, ps), dtype=np.float32)

        label_mask[~valid] = 0.0

        _, anf_raw = self._get_anf_patch(track_str, y0, x0, ps)
        if anf_raw is not None:
            label_mask[anf_raw >= self.max_label_anf] = 0.0

        # Augmentation: horizontal flip only
        if self.augment and np.random.random() > 0.5:
            pre_patch = pre_patch[:, :, ::-1].copy()
            post_patch = post_patch[:, :, ::-1].copy()
            label_mask = label_mask[:, ::-1].copy()
            if static is not None:
                static = static[:, :, ::-1].copy()
                static[2] = -static[2]  # negate aspect_easting (channel 2 in static)

        result = {
            'pre': torch.from_numpy(np.ascontiguousarray(pre_patch)),
            'post': torch.from_numpy(np.ascontiguousarray(post_patch)),
            'label': torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis])),
        }
        if static is not None:
            result['static'] = torch.from_numpy(np.ascontiguousarray(static))

        return result


def build_lazy_dataset(nc_path, date_polygon_pairs, sar_only=False, **kwargs):
    """Build a SiameseDebrisDataset using the same infrastructure as pairwise.

    Builds the base pairwise dataset for metadata, then wraps it as Siamese.
    """
    base = _build_lazy_dataset_base(
        nc_path, date_polygon_pairs, sar_only=sar_only, **kwargs,
    )
    # Convert to SiameseDebrisDataset by copying metadata
    ds = SiameseDebrisDataset.__new__(SiameseDebrisDataset)
    ds.__dict__.update(base.__dict__)
    ds.__class__ = SiameseDebrisDataset
    return ds
