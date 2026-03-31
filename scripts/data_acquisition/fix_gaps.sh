#!/bin/bash
# Fix remaining dataset gaps
# conda run -n sarvalanche bash scripts/data_acquisition/fix_gaps.sh 2>&1 | tee local/issw/fix_gaps_log.txt

SCRIPT="scripts/data_acquisition/prepare_center_datasets.py"

echo "=== Fix gaps: $(date) ==="

# 1. SNFAC/Sawtooth 24-25 — TV despeckle only
echo ""
echo "=== [1/4] SNFAC/Sawtooth 24-25 despeckle — $(date) ==="
python -c "
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf
from sarvalanche.preprocessing.pipelines import preprocess_rtc
from pathlib import Path
import glob

# Try both naming conventions
candidates = sorted(glob.glob('local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_*Sawtooth*.nc'))
for nc_path in candidates:
    nc = Path(nc_path)
    ds = load_netcdf_to_dataset(nc)
    if ds.attrs.get('preprocessed') == 'rtc_tv':
        print(f'{nc.name}: already preprocessed')
        ds.close()
        continue
    print(f'{nc.name}: running TV despeckle...')
    ds = preprocess_rtc(ds, tv_weight=0.5)
    ds.attrs['preprocessed'] = 'rtc_tv'
    export_netcdf(ds, nc, overwrite=True)
    print(f'{nc.name}: done')
    del ds
    import gc; gc.collect()
" || echo "[1/4] SNFAC DESPECKLE FAILED"

# 2. CAIC 24-25 — full rebuild (stub deleted)
echo ""
echo "=== [2/4] CAIC 24-25 rebuild — $(date) ==="
python $SCRIPT --center CAIC --include-zones "CAIC zone_2752" --out-dir local/issw/caic || echo "[2/4] CAIC REBUILD FAILED"

# 3. CNFAIC 24-25 — rebuild (corrupted nc deleted)
# 23-24 — build fresh without donor static_fp to avoid shape mismatch
echo ""
echo "=== [3/4] CNFAIC rebuild both winters — $(date) ==="
# Remove the 23-24 that doesn't exist anyway, and ensure no stale donor nc
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/issw_analysis')
from build_season_dataset import build_season_dataset, fetch_center_zones, season_nc_filename

zones = fetch_center_zones('CNFAIC')
aoi = zones['Turnagain Pass and Girdwood']['geometry']
zone_dir = Path('local/issw/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood')
zone_dir.mkdir(parents=True, exist_ok=True)

# 2023-2024: build fresh, NO static_fp (avoid shape mismatch)
nc_23 = season_nc_filename('2023-2024', 'Turnagain Pass and Girdwood')
print(f'Building {nc_23}...')
try:
    build_season_dataset(
        aoi=aoi,
        season_start='2023-11-01',
        season_end='2024-05-01',
        cache_dir=zone_dir,
        static_fp=None,
        track_gpkg=None,
        nc_filename=nc_23,
    )
    print(f'{nc_23}: done')
except Exception as e:
    print(f'{nc_23}: FAILED — {e}')

import gc; gc.collect()

# 2024-2025: rebuild
nc_24 = season_nc_filename('2024-2025', 'Turnagain Pass and Girdwood')
print(f'Building {nc_24}...')
try:
    build_season_dataset(
        aoi=aoi,
        season_start='2024-11-01',
        season_end='2025-05-01',
        cache_dir=zone_dir,
        static_fp=None,
        track_gpkg=zone_dir / 'season_tracks.gpkg' if (zone_dir / 'season_tracks.gpkg').exists() else None,
        nc_filename=nc_24,
    )
    print(f'{nc_24}: done')
except Exception as e:
    print(f'{nc_24}: FAILED — {e}')
" || echo "[3/4] CNFAIC REBUILD FAILED"

# 4. HRRR backfill on any datasets still missing t2m
echo ""
echo "=== [4/4] HRRR backfill sweep — $(date) ==="
python -c "
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf
from sarvalanche.io.hrrr import get_hrrr_for_dataset
from pathlib import Path
import gc

for nc in sorted(Path('local/issw').rglob('season_*_*.nc')):
    if 'v2_season' in str(nc) or 'v3_' in str(nc):
        continue
    try:
        ds = load_netcdf_to_dataset(nc)
    except Exception:
        print(f'CORRUPT: {nc}')
        continue
    if 't2m' in ds.data_vars:
        ds.close()
        continue
    if ds.attrs.get('preprocessed') != 'rtc_tv':
        ds.close()
        continue
    print(f'{nc.parent.name}/{nc.name}: fetching HRRR...')
    try:
        ds['t2m'] = get_hrrr_for_dataset(ds)
        export_netcdf(ds, nc, overwrite=True)
        print(f'  Done')
    except Exception as e:
        print(f'  FAILED: {e}')
    del ds; gc.collect()
print('HRRR sweep complete')
" || echo "[4/4] HRRR BACKFILL FAILED"

echo ""
echo "=== All fixes done: $(date) ==="

# Final inventory
echo ""
echo "=== Final inventory ==="
python -c "
from pathlib import Path
import xarray as xr

centers = {
    'uac': ['Salt_Lake','Logan','Uintas','Provo','Skyline','Ogden'],
    'snfac': ['Galena_Summit_&_Eastern_Mtns','Sawtooth_&_Western_Smoky_Mtns','Soldier_&_Wood_River_Valley_Mtns'],
    'caic': ['CAIC_zone_2752'],
    'fac': ['Flathead_Range_&_Glacier_NP'],
    'btac': ['Tetons'],
    'esac': ['Eastside_Region'],
    'cnfaic': ['Turnagain_Pass_and_Girdwood'],
    'nwac': ['Snoqualmie_Pass'],
}
winters = ['2023-2024', '2024-2025']
ok = 0
total = 0
for center, zones in centers.items():
    for zone in zones:
        for w in winters:
            total += 1
            nc = Path(f'local/issw/{center}/netcdfs/{zone}/season_{w}_{zone}.nc')
            if nc.exists() and nc.stat().st_size > 1e6:
                try:
                    ds = xr.open_dataset(nc)
                    pp = ds.attrs.get('preprocessed') == 'rtc_tv'
                    t2m = 't2m' in ds.data_vars
                    ds.close()
                    if pp and t2m:
                        ok += 1
                        continue
                    print(f'INCOMPLETE: {center}/{zone} {w} pp={pp} t2m={t2m}')
                except:
                    print(f'CORRUPT: {center}/{zone} {w}')
            else:
                print(f'MISSING: {center}/{zone} {w}')
print(f'\n{ok}/{total} fully complete')
"
