import numpy as np
import torch
from torch.utils.data import Dataset


class TrackPatchDataset(Dataset):
    """
    PyTorch Dataset for multi-channel raster patches of track polygons.

    Each item is a ``(C, H, W)`` float32 tensor and a scalar binary label.
    Channel order matches ``sarvalanche.ml.track_features.PATCH_CHANNELS``:
    distance_mahalanobis (0), p_empirical (1), fcf (2), slope (3),
    northing (4), easting (5), track_mask (6).

    Augmentation
    ------------
    Only horizontal flips are applied to preserve geographic orientation:
    - Horizontal flip (50% probability): negates easting channel (5) so that
      the coordinate grid remains consistent after mirroring.
    - Mild Gaussian noise on data channels 0-3.

    Parameters
    ----------
    patches : np.ndarray of shape (N, C, H, W)
        Pre-extracted patch arrays from ``extract_track_patch``.
    labels : np.ndarray of shape (N,)
        Binary labels (0 = no debris, 1 = debris).
    augment : bool
        Apply random augmentation at each sample draw.
    noise_std : float
        Std of Gaussian noise added to data channels 0-3. Set to 0 to disable.
    """

    _EASTING_CH: int = 5  # channel index to negate on horizontal flip

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.03,
    ):
        self.patches   = np.asarray(patches, dtype=np.float32)
        self.labels    = np.asarray(labels,  dtype=np.float32)
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        patch = self.patches[idx].copy()  # (C, H, W)
        label = self.labels[idx]

        if self.augment:
            if np.random.random() < 0.5:
                patch = patch[:, :, ::-1].copy()
                patch[self._EASTING_CH] = -patch[self._EASTING_CH]
            if self.noise_std > 0:
                patch[:4] += (
                    np.random.randn(*patch[:4].shape) * self.noise_std
                ).astype(np.float32)

        return torch.from_numpy(patch), torch.tensor(label)


class TrackSegDataset(Dataset):
    """
    PyTorch Dataset for segmentation training with pixel-wise soft targets.

    Each item is a ``(C, H, W)`` float32 input tensor, a ``(1, H, W)``
    float32 target map (from ``p_pixelwise``), and optionally a scalar binary
    label for joint classification training.

    Augmentation mirrors ``TrackPatchDataset``: horizontal flip (negates
    easting, flips target) and mild Gaussian noise on channels 0-3.

    Parameters
    ----------
    patches : np.ndarray of shape (N, C, H, W)
        Pre-extracted patch arrays from ``extract_track_patch``.
    targets : np.ndarray of shape (N, 1, H, W)
        Soft segmentation targets from ``extract_track_patch_with_target``.
    labels : np.ndarray of shape (N,), optional
        Binary labels for joint classification loss. If provided, ``__getitem__``
        returns ``(patch, target, label)``; otherwise ``(patch, target)``.
    augment : bool
        Apply random augmentation at each sample draw.
    noise_std : float
        Std of Gaussian noise added to data channels 0-3.
    """

    _EASTING_CH: int = 5

    def __init__(
        self,
        patches: np.ndarray,
        targets: np.ndarray,
        labels: np.ndarray | None = None,
        augment: bool = False,
        noise_std: float = 0.03,
    ):
        self.patches   = np.asarray(patches, dtype=np.float32)
        self.targets   = np.asarray(targets, dtype=np.float32)
        self.labels    = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch  = self.patches[idx].copy()   # (C, H, W)
        target = self.targets[idx].copy()   # (1, H, W)

        if self.augment:
            if np.random.random() < 0.5:
                patch  = patch[:, :, ::-1].copy()
                target = target[:, :, ::-1].copy()
                patch[self._EASTING_CH] = -patch[self._EASTING_CH]
            if self.noise_std > 0:
                patch[:4] += (
                    np.random.randn(*patch[:4].shape) * self.noise_std
                ).astype(np.float32)

        if self.labels is not None:
            return torch.from_numpy(patch), torch.from_numpy(target), torch.tensor(self.labels[idx])
        return torch.from_numpy(patch), torch.from_numpy(target)
