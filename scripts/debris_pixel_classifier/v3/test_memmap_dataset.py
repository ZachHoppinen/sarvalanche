"""Quick test: build memmap dataset, load a few batches."""
if __name__ != '__main__':
    raise SystemExit(0)

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.patch_extraction import build_static_stack, get_pair_metadata_and_tracks
from sarvalanche.ml.v3.dataset_inmemory import build_inmemory_dataset

nc = Path("local/issw/snfac/netcdfs") / "Sawtooth_&_Western_Smoky_Mtns" / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"

print("Loading dataset...")
t0 = time.time()
ds = load_netcdf_to_dataset(nc)
if not np.issubdtype(ds["time"].dtype, np.datetime64):
    ds["time"] = pd.DatetimeIndex(ds["time"].values)
ds = ds.load()
print(f"  Loaded in {time.time()-t0:.0f}s")

static_scene = build_static_stack(ds)
pair_metas, tracks, hrrr_cache = get_pair_metadata_and_tracks(ds)
print(f"  {len(pair_metas)} pairs, {len(tracks)} tracks")

# Just one date for speed
gdf = gpd.read_file("local/issw/debris_shapes/SNFAC/avalanche_labels_2025-02-04.gpkg")
gt_dir = Path("local/issw/debris_shapes/SNFAC/geotiffs/2025-02-04")
H, W = ds.sizes['y'], ds.sizes['x']
val_path_mask = np.zeros((H, W), dtype=bool)

print("Building dataset...")
t0 = time.time()
dataset = build_inmemory_dataset(
    ds, pair_metas, tracks, hrrr_cache, static_scene,
    [("2025-02-04", gdf, gt_dir)], val_path_mask,
    stride=128, neg_ratio=1.0,
)
print(f"  Built in {time.time()-t0:.0f}s, {len(dataset)} samples")

# Test single item
print("\nTesting __getitem__...")
t0 = time.time()
item = dataset[0]
print(f"  Item 0: x={item['x'].shape}, label={item['label'].shape}, conf={item['confidence']:.2f}")
print(f"  Time: {time.time()-t0:.3f}s")

# Test 100 items
t0 = time.time()
for i in range(100):
    _ = dataset[i]
print(f"  100 items: {time.time()-t0:.2f}s ({(time.time()-t0)/100*1000:.1f}ms/item)")

# Test DataLoader with 0 workers
print("\nDataLoader num_workers=0...")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 5:
        break
print(f"  5 batches: {time.time()-t0:.2f}s")

# Test DataLoader with 2 workers
print("\nDataLoader num_workers=2...")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 5:
        break
print(f"  5 batches: {time.time()-t0:.2f}s")

# Test DataLoader with 4 workers
print("\nDataLoader num_workers=4...")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 5:
        break
print(f"  5 batches: {time.time()-t0:.2f}s")

print("\nDone!")
