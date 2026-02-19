import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr

class SARTimeSeriesDataset(Dataset):
    """
    Dataset for self-supervised SAR time series learning.

    Creates random sequences of T baseline images + 1 target image.
    """
    def __init__(self, sar_timeseries, min_seq_len=2, max_seq_len=10, patch_size=16):
        """
        Parameters
        ----------
        sar_timeseries : xr.DataArray or list of xr.DataArray
            SAR backscatter with dimensions (time, polarization, y, x) or (time, y, x)
            If list, will sample from multiple scenes
        min_seq_len : int
            Minimum baseline sequence length (default: 2)
        max_seq_len : int
            Maximum baseline sequence length (default: 10)
        patch_size : int
            Size of spatial patches to extract (default: 16)
        """
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size

        # Handle single DataArray or list
        if isinstance(sar_timeseries, xr.DataArray):
            self.scenes = [sar_timeseries]
        else:
            self.scenes = sar_timeseries

        # Pre-compute valid patches for each scene
        self.valid_patches = []
        for scene_idx, scene in enumerate(self.scenes):
            T, H, W = scene.shape[0], scene.shape[-2], scene.shape[-1]

            # Need at least min_seq_len + 1 timesteps
            if T < min_seq_len + 1:
                continue

            # Calculate number of patches
            n_patches_h = H // patch_size
            n_patches_w = W // patch_size

            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    self.valid_patches.append({
                        'scene_idx': scene_idx,
                        'patch_i': i,
                        'patch_j': j,
                    })

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        scene = self.scenes[patch_info['scene_idx']]

        # Random sequence length
        T_total = len(scene.time)
        T_baseline = np.random.randint(
            self.min_seq_len,
            min(self.max_seq_len, T_total - 1) + 1
        )

        # Random starting point
        max_start = T_total - T_baseline - 1  # need at least 1 timestep after baseline for target
        if max_start < 0:
            # fallback: use all available as baseline, last step as target
            T_baseline = T_total - 1
            max_start  = 0

        t_start = np.random.randint(0, max_start + 1)
        t_end   = t_start + T_baseline

        # Extract patch
        i = patch_info['patch_i']
        j = patch_info['patch_j']
        ps = self.patch_size

        y_start = i * ps
        x_start = j * ps

        # Get baseline and target
        baseline = scene.isel(
            time=slice(t_start, t_end),
            y=slice(y_start, y_start + ps),
            x=slice(x_start, x_start + ps)
        ).values  # (T_baseline, 2, 16, 16) or (T_baseline, 16, 16)

        target = scene.isel(
            time=t_end,
            y=slice(y_start, y_start + ps),
            x=slice(x_start, x_start + ps)
        ).values  # (2, 16, 16) or (16, 16)

        # Ensure channel dimension exists
        if baseline.ndim == 3:  # (T, H, W)
            baseline = baseline[:, None, :, :]  # (T, 1, H, W)
        if target.ndim == 2:  # (H, W)
            target = target[None, :, :]  # (1, H, W)

        # Handle NaNs (replace with 0 or skip - here we replace)
        baseline = np.nan_to_num(baseline, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        return {
            'baseline': torch.FloatTensor(baseline),  # (T, C, H, W)
            'target': torch.FloatTensor(target),      # (C, H, W)
        }
