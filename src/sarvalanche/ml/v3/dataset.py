"""PyTorch Dataset for v3 single-pair CNN training.

Each .npz contains one crossing pair:
  - sar: (N_SAR, 128, 128) — 7 SAR channels for one pair
  - static: (N_STATIC, 128, 128) — 13 static terrain channels
  - label_mask: (128, 128) — pixel-level debris label (0/1)
  - label: int8 (0 or 1, patch-level)
  - val_path_mask: (128, 128) — True where AKDOT/AKRR validation paths exist
    (these pixels are excluded from training loss but used for validation)

Training: patches where val_path_mask is all False (no overlap with val paths)
Validation: patches where val_path_mask has any True pixels (held out)
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sarvalanche.ml.v3.channels import N_SAR, N_STATIC

log = logging.getLogger(__name__)


class V3PairDataset(Dataset):
    """Dataset loading v3 single-pair .npz patches.

    Parameters
    ----------
    data_dir : Path
        Directory containing .npz files and labels.json.
    augment : bool
        Apply data augmentation (horizontal flip + noise).
    split : str
        'train' = patches with no validation path overlap
        'val' = patches with validation path overlap
        'all' = everything
    """

    def __init__(self, data_dir: Path, augment: bool = True, split: str = 'all'):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.split = split

        self.files: list[Path] = []
        self.labels: list[int] = []
        self.confidences: list[float] = []
        self.has_val_path: list[bool] = []

        # Discover patches from labels.json
        for labels_path in sorted(self.data_dir.rglob('labels.json')):
            with open(labels_path) as f:
                labels_data = json.load(f)

            parent = labels_path.parent
            for patch_id, meta in labels_data.items():
                label = meta.get('label')
                if label not in (0, 1):
                    continue

                # Find all per-pair npz files for this patch
                pair_files = sorted(parent.glob(f'{patch_id}_v3_pair*.npz'))
                if not pair_files:
                    # Fallback: try v3_ prefix
                    pair_files = sorted(parent.glob(f'{patch_id}_v3_*.npz'))

                confidence = meta.get('confidence', 1.0)
                on_val_path = meta.get('on_val_path', False)

                for pf in pair_files:
                    self.files.append(pf)
                    self.labels.append(label)
                    self.confidences.append(confidence)
                    self.has_val_path.append(on_val_path)

        # Apply split filter
        if split == 'train':
            mask = [not v for v in self.has_val_path]
            self._apply_mask(mask)
        elif split == 'val':
            mask = list(self.has_val_path)
            self._apply_mask(mask)

        log.info(
            'V3PairDataset(%s): %d patches (%d debris, %d no-debris) from %s',
            split, len(self.files), sum(self.labels),
            len(self.labels) - sum(self.labels), self.data_dir,
        )

    def _apply_mask(self, mask):
        self.files = [f for f, m in zip(self.files, mask) if m]
        self.labels = [l for l, m in zip(self.labels, mask) if m]
        self.confidences = [c for c, m in zip(self.confidences, mask) if m]
        self.has_val_path = [v for v, m in zip(self.has_val_path, mask) if m]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=True)
        sar = data['sar'].astype(np.float32)          # (N_SAR, H, W)
        static = data['static'].astype(np.float32)    # (N_STATIC, H, W)

        # Label mask
        if 'label_mask' in data:
            label_mask = data['label_mask'].astype(np.float32)  # (H, W)
        else:
            label = int(data['label'])
            label_mask = np.full(sar.shape[-2:], label, dtype=np.float32)

        # Validation path mask
        if 'val_path_mask' in data:
            val_mask = data['val_path_mask'].astype(np.float32)  # (H, W)
        else:
            val_mask = np.zeros(sar.shape[-2:], dtype=np.float32)

        confidence = self.confidences[idx]

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                sar = sar[:, :, ::-1].copy()
                static = static[:, :, ::-1].copy()
                label_mask = label_mask[:, ::-1].copy()
                val_mask = val_mask[:, ::-1].copy()
            # Small noise on SAR change channel only
            sar[0] += np.random.randn(*sar[0].shape).astype(np.float32) * 0.05

        # Concatenate SAR + static → single input tensor
        x = np.concatenate([sar, static], axis=0)  # (N_INPUT, H, W)

        return {
            'x': torch.from_numpy(np.ascontiguousarray(x)),
            'label': torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis])),
            'confidence': torch.tensor(confidence, dtype=torch.float32),
            'val_mask': torch.from_numpy(np.ascontiguousarray(val_mask[np.newaxis])),
        }
