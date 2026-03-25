"""PyTorch Dataset for v2 CNN training patches.

Reads .npz patches saved by patch_labeler.py --v2. Each .npz contains:
  - sar_maps: (N, 4, 128, 128) per-track/pol [change, ANF, anomaly, edges]
  - static: (6, 128, 128) normalized static terrain channels
  - label: int8 (0 or 1)

Labels come from a labels.json that maps window IDs to metadata including
the binary label and v2 tile filenames.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


log = logging.getLogger(__name__)


class V2PatchDataset(Dataset):
    """Dataset loading v2 format .npz patches for two-pass CNN training.

    Parameters
    ----------
    data_dir : Path
        Directory containing .npz patch files and labels.json.
    augment : bool
        Whether to apply data augmentation (horizontal flip + noise).
    """

    def __init__(self, data_dir: Path, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment

        # Discover all .npz files with v2 format
        self.files: list[Path] = []
        self.labels: list[int] = []

        self.confidences: list[float] = []

        # Load labels from all labels.json files in subdirectories
        for labels_path in sorted(self.data_dir.rglob('labels.json')):
            with open(labels_path) as f:
                labels_data = json.load(f)

            parent = labels_path.parent
            for window_id, meta in labels_data.items():
                label = meta.get('label')
                if label not in (0, 1):
                    continue
                confidence = meta.get('confidence', 1.0)
                # Find v2 tiles for this window
                v2_tiles = sorted(parent.glob(f'{window_id}_v2_*.npz'))
                for tile_path in v2_tiles:
                    if tile_path.exists():
                        self.files.append(tile_path)
                        self.labels.append(label)
                        self.confidences.append(confidence)

        if not self.files:
            # Fallback: load all v2 .npz files directly
            for npz_path in sorted(self.data_dir.rglob('*_v2_*.npz')):
                data = np.load(npz_path, allow_pickle=True)
                if 'sar_maps' in data and 'static' in data and 'label' in data:
                    self.files.append(npz_path)
                    self.labels.append(int(data['label']))
                    self.confidences.append(1.0)

        log.info(
            'V2PatchDataset: %d patches (%d debris, %d no-debris) from %s',
            len(self.files),
            sum(self.labels),
            len(self.labels) - sum(self.labels),
            self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        confidence = self.confidences[idx] if idx < len(self.confidences) else 1.0
        data = np.load(self.files[idx], allow_pickle=True)
        sar_maps = data['sar_maps'].astype(np.float32)
        if sar_maps.ndim == 2:
            # Single map: (H, W) → (1, H, W)
            sar_maps = sar_maps[np.newaxis]
        if sar_maps.ndim == 3:
            # Legacy format: (N, H, W) — no ANF/anomaly/edge channels
            # Apply log1p compression and pad to 4 channels
            sar_maps = np.sign(sar_maps) * np.log1p(np.abs(sar_maps))
            ones = np.ones_like(sar_maps)
            zeros = np.zeros_like(sar_maps)
            # (N, 4, H, W): [change, ANF=1, anomaly=0, edges=0]
            sar_maps = np.stack([sar_maps, ones, zeros, zeros], axis=1)
        # sar_maps is now (N, C, H, W) where C is 2 or 4
        # 4-channel patches already have all channels from build_v2_patch
        static = data['static'].astype(np.float32)       # (N_STATIC, 128, 128)
        label = int(data['label'])

        # Use pixel-level label mask if available, else broadcast window label
        if 'label_mask' in data:
            label_mask = data['label_mask'].astype(np.float32)  # (128, 128)
        else:
            label_mask = np.full(sar_maps.shape[-2:], label, dtype=np.float32)

        # Augmentation: random horizontal flip (flip last axis = W)
        if self.augment and np.random.random() > 0.5:
            sar_maps = sar_maps[:, :, :, ::-1].copy()
            static = static[:, :, ::-1].copy()
            label_mask = label_mask[:, ::-1].copy()

        # Augmentation: small additive noise on SAR change channel only (not ANF)
        if self.augment:
            noise_scale = 0.05
            for i in range(len(sar_maps)):
                sar_maps[i, 0] += np.random.randn(*sar_maps[i, 0].shape).astype(np.float32) * noise_scale

        # Convert each SAR map to individual (2, H, W) tensors
        sar_list = [torch.from_numpy(np.ascontiguousarray(sar_maps[i])) for i in range(len(sar_maps))]

        # Label as (1, H, W)
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_mask[np.newaxis]))

        return {
            'sar_maps': sar_list,
            'static': torch.from_numpy(np.ascontiguousarray(static)),
            'label': label_tensor,
            'confidence': torch.tensor(confidence, dtype=torch.float32),
        }


def v2_collate_fn(batch: list[dict]) -> dict:
    """Custom collate for variable-length SAR map lists.

    All samples in a batch must have the same number of SAR maps
    (pad with zeros if needed at the dataset level).

    Returns:
    -------
    dict with:
        sar_maps: list of (B, 2, H, W) tensors, one per track/pol slot
        static: (B, C, H, W)
        label: (B, 1, H, W)
    """
    # Find max number of SAR maps in batch
    max_n = max(len(item['sar_maps']) for item in batch)

    # Pad samples with fewer maps using zero tensors
    sar_lists: list[list[torch.Tensor]] = []
    for item in batch:
        maps = item['sar_maps']
        while len(maps) < max_n:
            maps.append(torch.zeros_like(maps[0]))
        sar_lists.append(maps)

    # Stack per slot: list of N tensors, each (B, 1, H, W)
    sar_maps = [
        torch.stack([sar_lists[b][i] for b in range(len(batch))])
        for i in range(max_n)
    ]

    static = torch.stack([item['static'] for item in batch])
    label = torch.stack([item['label'] for item in batch])
    confidence = torch.stack([item['confidence'] for item in batch])

    return {
        'sar_maps': sar_maps,
        'static': static,
        'label': label,
        'confidence': confidence,
    }
