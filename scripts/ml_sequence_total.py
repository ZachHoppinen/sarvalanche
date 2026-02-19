"""
Estimate total number of training sequences available from cached scene NetCDFs.

Mirrors the logic in SARTimeSeriesDataset to give a realistic sequence count.
"""

from pathlib import Path
import xarray as xr
import numpy as np
from collections import defaultdict

# --- CONFIG ---
SCENE_CACHE_DIR = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data/scene_cache')
PATCH_SIZE      = 16
STRIDE          = 48   # match SARTimeSeriesDataset — change to 16 for no stride
MIN_SEQ_LEN     = 7
MAX_SEQ_LEN     = 10

def count_sequences_for_scene(nc_path, patch_size=16, stride=None, min_seq_len=2, max_seq_len=10):
    """
    For a single NetCDF scene, estimate the number of valid training sequences.

    For each patch:
      - T_total timesteps → valid t_start values for each possible T_baseline
      - Sum over all T_baseline in [min_seq_len, min(max_seq_len, T_total-1)]
        of (T_total - T_baseline) possible starting points
    """
    stride = stride if stride is not None else patch_size  # default: non-overlapping

    with xr.open_dataset(nc_path) as ds:
        T = len(ds.time)
        H = ds['y'].size
        W = ds['x'].size

    if T < min_seq_len + 1:
        return 0, T, H, W

    # Mirror SARTimeSeriesDataset stride logic exactly
    n_patches_h = len(range(0, H - patch_size + 1, stride))
    n_patches_w = len(range(0, W - patch_size + 1, stride))
    n_patches   = n_patches_h * n_patches_w

    # For each patch, count valid (t_start, T_baseline) combinations
    sequences_per_patch = 0
    for t_baseline in range(min_seq_len, min(max_seq_len, T - 1) + 1):
        max_start = T - t_baseline - 1
        sequences_per_patch += max_start + 1  # number of valid t_start values

    total_sequences = n_patches * sequences_per_patch
    return total_sequences, T, H, W, n_patches


def main():
    nc_files = sorted(SCENE_CACHE_DIR.glob('*__*__*.nc'))

    if not nc_files:
        print(f"No .nc files found in {SCENE_CACHE_DIR}")
        return

    print(f"Found {len(nc_files)} scene files")
    print(f"Config: patch_size={PATCH_SIZE}, stride={STRIDE}, "
          f"seq_len=[{MIN_SEQ_LEN},{MAX_SEQ_LEN}]\n")
    print(f"{'File':<60} {'T':>4} {'H':>5} {'W':>5} {'Patches':>8} {'Sequences':>12}")
    print('-' * 100)

    total_sequences = 0
    total_patches   = 0
    by_center       = defaultdict(int)
    by_season       = defaultdict(int)
    skipped         = []

    for nc_path in nc_files:
        try:
            result = count_sequences_for_scene(
                nc_path, PATCH_SIZE, STRIDE, MIN_SEQ_LEN, MAX_SEQ_LEN
            )
            n_seq, T, H, W, n_patches = result

            if n_seq == 0:
                skipped.append((nc_path.name, T))
                continue

            print(f"{nc_path.name:<60} {T:>4} {H:>5} {W:>5} {n_patches:>8,} {n_seq:>12,}")

            total_sequences += n_seq
            total_patches   += n_patches

            # Breakdown by center and season
            parts     = nc_path.stem.split('__')
            center_id = parts[0].split('_')[0]
            season    = parts[-1]  # e.g. '2019-12'

            by_center[center_id] += n_seq
            by_season[season]    += n_seq

        except Exception as e:
            print(f"  ERROR reading {nc_path.name}: {e}")

    print('-' * 100)
    print(f"\n{'TOTAL':.<60} {total_patches:>18,} patches")
    print(f"{'TOTAL sequences':.<60} {total_sequences:>18,}")
    print(f"\n  (Paper reports 2,511,348 sequences for comparison)")
    print(f"  Your dataset is {100 * total_sequences / 2_511_348:.1f}% of paper size")

    print(f"\n--- By center ---")
    for center, n in sorted(by_center.items(), key=lambda x: -x[1]):
        print(f"  {center:<20} {n:>12,}  ({100*n/max(total_sequences,1):.1f}%)")

    print(f"\n--- By season ---")
    for season, n in sorted(by_season.items()):
        print(f"  {season:<20} {n:>12,}  ({100*n/max(total_sequences,1):.1f}%)")

    if skipped:
        print(f"\n--- Skipped (< {MIN_SEQ_LEN+1} timesteps) ---")
        for name, T in skipped:
            print(f"  {name}  (T={T})")

    print(f"\n--- Batch estimate ---")
    for batch_size in [32, 256]:
        batches_per_epoch = total_sequences // batch_size
        print(f"  batch_size={batch_size:<4}  → {batches_per_epoch:>8,} batches/epoch")


if __name__ == '__main__':
    main()