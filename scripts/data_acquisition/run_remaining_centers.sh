#!/bin/bash
# Sequential dataset preparation for remaining gaps
# Run with: conda run -n sarvalanche bash scripts/data_acquisition/run_remaining_centers.sh

SCRIPT="scripts/data_acquisition/prepare_center_datasets.py"
BASE="local/issw"

echo "=== Starting remaining builds: $(date) ==="
echo "=== Base directory: $BASE ==="

# SAC (both winters needed)
echo ""
echo "=== SAC (Central Sierra Nevada) — $(date) ==="
python $SCRIPT --center SAC --out-dir "$BASE/sac" || echo "SAC FAILED"

# NWAC (both winters needed)
echo ""
echo "=== NWAC (Snoqualmie Pass) — $(date) ==="
python $SCRIPT --center NWAC --include-zones "Snoqualmie Pass" --out-dir "$BASE/nwac" || echo "NWAC FAILED"

# ESAC 2024-2025 (2023-2024 already done)
echo ""
echo "=== ESAC (Eastside Region) — $(date) ==="
python $SCRIPT --center ESAC --out-dir "$BASE/esac" || echo "ESAC FAILED"

# CAIC 2024-2025 (2023-2024 already done)
echo ""
echo "=== CAIC (zone 2752) — $(date) ==="
python $SCRIPT --center CAIC --include-zones "CAIC zone_2752" --out-dir "$BASE/caic" || echo "CAIC FAILED"

# CNFAIC 2023-2024 (2024-2025 already exists)
echo ""
echo "=== CNFAIC (Turnagain Pass) — $(date) ==="
python $SCRIPT --center CNFAIC --include-zones "Turnagain Pass and Girdwood" --out-dir "$BASE/cnfaic" || echo "CNFAIC FAILED"

# Backfill HRRR on CNFAIC 2024-2025
echo ""
echo "=== Backfill HRRR on CNFAIC 2024-2025 — $(date) ==="
python -c "
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf
from sarvalanche.io.hrrr import get_hrrr_for_dataset
from pathlib import Path
nc = Path('$BASE/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2024-2025_Turnagain_Pass_and_Girdwood.nc')
if not nc.exists():
    print(f'File not found: {nc}')
else:
    ds = load_netcdf_to_dataset(nc)
    if 't2m' not in ds.data_vars:
        print('Fetching HRRR for CNFAIC 2024-2025...')
        ds['t2m'] = get_hrrr_for_dataset(ds)
        ds = ds.load()
        export_netcdf(ds, nc, overwrite=True)
        print('Done')
    else:
        print('Already has t2m')
" || echo "CNFAIC HRRR BACKFILL FAILED"

# Backfill HRRR on old SNFAC datasets
echo ""
echo "=== Backfill HRRR on SNFAC — $(date) ==="
python -c "
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf
from sarvalanche.io.hrrr import get_hrrr_for_dataset
from pathlib import Path
import gc

for nc in sorted(Path('$BASE/snfac/netcdfs').rglob('season_*_*.nc')):
    if 'v2_season' in str(nc) or 'v3_' in str(nc):
        continue
    ds = load_netcdf_to_dataset(nc)
    if 't2m' in ds.data_vars:
        print(f'{nc.parent.name}/{nc.name}: already has t2m')
        ds.close()
        continue
    print(f'{nc.parent.name}/{nc.name}: fetching HRRR...')
    try:
        ds['t2m'] = get_hrrr_for_dataset(ds)
        ds = ds.load()
        export_netcdf(ds, nc, overwrite=True)
        print(f'  Done')
    except Exception as e:
        print(f'  FAILED: {e}')
    del ds
    gc.collect()
" || echo "SNFAC HRRR BACKFILL FAILED"

echo ""
echo "=== All done: $(date) ==="

# Summary
echo ""
echo "=== Run log summary ==="
for center in sac nwac esac caic cnfaic; do
    log="$BASE/$center/prepare_run_log.csv"
    if [ -f "$log" ]; then
        echo "--- $center ---"
        cat "$log"
    fi
done

echo ""
echo "=== Dataset inventory ==="
find "$BASE" -name "season_*_*.nc" ! -path "*/v2_*" ! -path "*/v3_*" -exec ls -lh {} \; | awk '{print $5, $9}' | sort -k2
