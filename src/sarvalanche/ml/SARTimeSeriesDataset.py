import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm.auto import tqdm
import zarr


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
        # self._cache = {}  # per-worker file cache
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

    # def _preload(self):
    #     unique_scenes = list(dict.fromkeys(p['scene_idx'] for p in self.valid_patches))
    #     for idx in tqdm(unique_scenes, desc='Preloading scenes'):
    #         scene = self.scenes[idx]
    #         if isinstance(scene, Path):
    #             self._load_scene(scene)  # populates cache as a side effect

    # def _open(self, path: Path):
    #     """Open a zarr or netCDF file as a dataset."""
    #     if path.suffix == '.zarr':
    #         return xr.open_zarr(path, consolidated=False)
    #     return xr.open_dataset(path)

    def _get_shape(self, scene):
        scene = self._load_scene(scene)
        return scene.shape[0], scene.shape[-2], scene.shape[-1]

    # def _load_scene(self, scene):
    #     if isinstance(scene, Path):
    #         if scene not in self._cache:
    #             self._cache[scene] = np.load(scene, mmap_mode='r')
    #         return self._cache[scene]
    #     return scene

    def _load_scene(self, scene):
        if isinstance(scene, Path):
            if scene.suffix == '.zarr':
                return zarr.open(str(scene), mode='r')['backscatter']
            return np.load(scene, mmap_mode='r')
        return scene
    # def _ds_to_array(self, ds):
    #     if 'backscatter' in ds:
    #         return ds['backscatter']
    #     return xr.concat([ds['VV'], ds['VH']], dim='polarization') \
    #             .transpose('time', 'polarization', 'y', 'x')

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        scene      = self._load_scene(self.scenes[patch_info['scene_idx']])

        T_total    = scene.shape[0]  # was len(scene.time)
        T_baseline = np.random.randint(self.min_seq_len, min(self.max_seq_len, T_total - 1) + 1)

        max_start = T_total - T_baseline - 1
        if max_start < 0:
            T_baseline = T_total - 1
            max_start  = 0

        t_start = np.random.randint(0, max_start + 1)
        t_end   = t_start + T_baseline

        y_start = patch_info['patch_y']
        x_start = patch_info['patch_x']

        baseline = scene[t_start:t_end, :, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        target   = scene[t_end,         :, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        baseline = np.nan_to_num(baseline, nan=0.0)
        target   = np.nan_to_num(target,   nan=0.0)

        # with self._counter.get_lock():
        #     self._counter.value += 1


        return {
            'baseline': torch.FloatTensor(baseline),
            'target':   torch.FloatTensor(target),
        }