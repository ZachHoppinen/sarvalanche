"""Minimal test: memmap + DataLoader workers on macOS Python 3.14."""
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork')

    import numpy as np
    import tempfile
    import time
    import torch
    from torch.utils.data import Dataset, DataLoader

    # Create memmap
    path = tempfile.mktemp(suffix='.dat')
    shape = (100, 2, 1000, 1000)
    print(f"Writing memmap {shape}...")
    fp = np.memmap(path, dtype=np.float32, mode='w+', shape=shape)
    fp[:] = np.random.randn(*shape).astype(np.float32)
    fp.flush()
    del fp
    print(f"  Done: {path}")

    class SimpleDataset(Dataset):
        def __init__(self, memmap_path, shape):
            self.memmap_path = memmap_path
            self.shape = shape
            self._data = None  # lazy open

        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            if self._data is None:
                self._data = np.memmap(self.memmap_path, dtype=np.float32,
                                       mode='r', shape=self.shape)
            pair_idx = idx % self.shape[0]
            patch = self._data[pair_idx, 0, 100:228, 100:228].copy()
            return {'x': torch.from_numpy(patch)}

    ds = SimpleDataset(path, shape)

    # Test single item
    print(f"\n__getitem__(0): {ds[0]['x'].shape}")

    # Test 0 workers
    print("\nnum_workers=0...")
    loader = DataLoader(ds, batch_size=32, num_workers=0)
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 5: break
    print(f"  5 batches: {time.time()-t0:.2f}s")

    # Test 2 workers
    print("\nnum_workers=2...")
    loader = DataLoader(ds, batch_size=32, num_workers=2)
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 5: break
    print(f"  5 batches: {time.time()-t0:.2f}s")

    # Test 4 workers
    print("\nnum_workers=4...")
    loader = DataLoader(ds, batch_size=32, num_workers=4)
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 5: break
    print(f"  5 batches: {time.time()-t0:.2f}s")

    import os
    os.remove(path)
    print("\nDone! Workers work.")
