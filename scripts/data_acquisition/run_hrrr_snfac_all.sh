#!/bin/bash
# Sequential HRRR temperature fetch for all SNFAC zones/seasons
# Run with: nohup bash scripts/data_acquisition/run_hrrr_snfac_all.sh > hrrr_snfac_all.log 2>&1 &

set -e

SCRIPT="scripts/data_acquisition/fetch_hrrr_temperature.py"
BASE="local/issw/snfac/netcdfs"
OUT="local/issw/snfac"

echo "=== Starting HRRR SNFAC batch: $(date) ==="

# Sawtooth 2021-2022
echo "--- Sawtooth 2021-2022: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --out "$OUT/hrrr_temperature_sawtooth_2122.nc" \
    --model hrrr

# Sawtooth 2022-2023
echo "--- Sawtooth 2022-2023: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --out "$OUT/hrrr_temperature_sawtooth_2223.nc" \
    --model hrrr

# Sawtooth 2023-2024
echo "--- Sawtooth 2023-2024: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --out "$OUT/hrrr_temperature_sawtooth_2324.nc" \
    --model hrrr

# Galena 2021-2022
echo "--- Galena 2021-2022: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Galena_Summit_&_Eastern_Mtns/season_2021-2022_Galena_Summit_&_Eastern_Mtns.nc" \
    --out "$OUT/hrrr_temperature_galena_2122.nc" \
    --model hrrr

# Galena 2022-2023
echo "--- Galena 2022-2023: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc" \
    --out "$OUT/hrrr_temperature_galena_2223.nc" \
    --model hrrr

# Galena 2023-2024
echo "--- Galena 2023-2024: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc" \
    --out "$OUT/hrrr_temperature_galena_2324.nc" \
    --model hrrr

# Galena 2024-2025
echo "--- Galena 2024-2025: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Galena_Summit_&_Eastern_Mtns/season_2024-2025_Galena_Summit_&_Eastern_Mtns.nc" \
    --out "$OUT/hrrr_temperature_galena_2425.nc" \
    --model hrrr

# Banner 2022-2023
echo "--- Banner 2022-2023: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Banner_Summit/season_2022-2023_Banner_Summit.nc" \
    --out "$OUT/hrrr_temperature_banner_2223.nc" \
    --model hrrr

# Banner 2023-2024
echo "--- Banner 2023-2024: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Banner_Summit/season_2023-2024_Banner_Summit.nc" \
    --out "$OUT/hrrr_temperature_banner_2324.nc" \
    --model hrrr

# Banner 2024-2025
echo "--- Banner 2024-2025: $(date) ---"
conda run -n sarvalanche python "$SCRIPT" \
    --nc "$BASE/Banner_Summit/season_2024-2025_Banner_Summit.nc" \
    --out "$OUT/hrrr_temperature_banner_2425.nc" \
    --model hrrr

echo "=== All done: $(date) ==="
