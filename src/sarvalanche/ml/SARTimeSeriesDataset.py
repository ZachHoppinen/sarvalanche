import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import numpy as np
import xarray as xr
from pathlib import Path


class SARTimeSeriesDataset(Dataset):
    """
    Dataset for self-supervised SAR time series learning.
    Accepts either xr.DataArray objects (in-memory) or Path objects (lazy disk reads).
    """
    def __init__(self, sar_timeseries, min_seq_len=2, max_seq_len=10, patch_size=16, stride=None):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.patch_size  = patch_size
        self.stride      = stride if stride is not None else patch_size  # default: non-overlapping
        self._cache = {}  # per-worker file cache
        # self._counter = mp.Value('i', 0)  # shared integer across workers


        # Normalise input to a list
        if isinstance(sar_timeseries, (xr.DataArray, Path)):
            sar_timeseries = [sar_timeseries]
        self.scenes = sar_timeseries  # list of DataArray or Path, can be mixed

        # Build patch index
        self.valid_patches = []
        for scene_idx, scene in enumerate(self.scenes):
            T, H, W = self._get_shape(scene)

            if T < min_seq_len + 1:
                print(f"  Skipping scene {scene_idx}: only {T} timesteps")
                continue

            for y in range(0, H - patch_size + 1, self.stride):
                for x in range(0, W - patch_size + 1, self.stride):
                    self.valid_patches.append({
                        'scene_idx': scene_idx,
                        'patch_y':   y,   # now absolute pixel coords, not patch index
                        'patch_x':   x,
                    })

    def _get_shape(self, scene):
        """Get (T, H, W) without loading data for Path inputs."""
        if isinstance(scene, Path):
            with xr.open_dataset(scene) as ds:
                da = self._ds_to_array(ds)
                return da.shape[0], da.shape[-2], da.shape[-1]
        else:
            return scene.shape[0], scene.shape[-2], scene.shape[-1]

    def _load_scene(self, scene):
        if isinstance(scene, Path):
            if scene not in self._cache:
                ds = xr.open_dataset(scene)
                self._cache[scene] = self._ds_to_array(ds)  # lazy DataArray
            return self._cache[scene]
        return scene


    def _ds_to_array(self, ds):
        """Stack VV+VH into (time, polarization, y, x)."""
        return xr.concat([ds['VV'], ds['VH']], dim='polarization') \
                 .transpose('time', 'polarization', 'y', 'x')

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        scene      = self._load_scene(self.scenes[patch_info['scene_idx']])

        T_total    = len(scene.time)
        T_baseline = np.random.randint(self.min_seq_len, min(self.max_seq_len, T_total - 1) + 1)

        max_start = T_total - T_baseline - 1
        if max_start < 0:
            T_baseline = T_total - 1
            max_start  = 0

        t_start = np.random.randint(0, max_start + 1)
        t_end   = t_start + T_baseline

        y_start = patch_info['patch_y']
        x_start = patch_info['patch_x']

        baseline = scene.isel(
            time=slice(t_start, t_end),
            y=slice(y_start, y_start + self.patch_size),
            x=slice(x_start, x_start + self.patch_size)
        ).values

        target = scene.isel(
            time=t_end,
            y=slice(y_start, y_start + self.patch_size),
            x=slice(x_start, x_start + self.patch_size)
        ).values

        # Ensure channel dim exists for single-pol case
        if baseline.ndim == 3:
            baseline = baseline[:, None, :, :]
        if target.ndim == 2:
            target = target[None, :, :]

        baseline = np.nan_to_num(baseline, nan=0.0)
        target   = np.nan_to_num(target,   nan=0.0)

        # with self._counter.get_lock():
        #     self._counter.value += 1

        return {
            'baseline': torch.FloatTensor(baseline),
            'target':   torch.FloatTensor(target),
        }