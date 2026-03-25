#!/bin/bash
# Generate auto-labels for all SNFAC zones/seasons with HRRR data.
# Run with: nohup bash scripts/debris_pixel_classifier/v3/run_autolabels_all.sh > autolabels_all.log 2>&1 &

set -e

SCRIPT="scripts/debris_pixel_classifier/v3/generate_autolabels.py"
NC_BASE="local/issw/snfac/netcdfs"
HRRR_BASE="local/issw/snfac"
OUT_BASE="local/issw/debris_shapes/snfac/autolabels"

echo "=== Starting autolabel batch: $(date) ==="

# Sawtooth 2021-2022
echo "--- Sawtooth 2021-2022: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_sawtooth_2122.nc" \
    --out-dir "$OUT_BASE/sawtooth_2122"

# Sawtooth 2022-2023
echo "--- Sawtooth 2022-2023: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_sawtooth_2223.nc" \
    --out-dir "$OUT_BASE/sawtooth_2223"

# Sawtooth 2023-2024
echo "--- Sawtooth 2023-2024: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_sawtooth_2324.nc" \
    --out-dir "$OUT_BASE/sawtooth_2324"

# Galena 2021-2022
echo "--- Galena 2021-2022: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Galena_Summit_&_Eastern_Mtns/season_2021-2022_Galena_Summit_&_Eastern_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_galena_2122.nc" \
    --out-dir "$OUT_BASE/galena_2122"

# Galena 2022-2023
echo "--- Galena 2022-2023: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_galena_2223.nc" \
    --out-dir "$OUT_BASE/galena_2223"

# Galena 2023-2024
echo "--- Galena 2023-2024: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_galena_2324.nc" \
    --out-dir "$OUT_BASE/galena_2324"

# Galena 2024-2025
echo "--- Galena 2024-2025: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Galena_Summit_&_Eastern_Mtns/season_2024-2025_Galena_Summit_&_Eastern_Mtns.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_galena_2425.nc" \
    --out-dir "$OUT_BASE/galena_2425"

# Banner 2022-2023
echo "--- Banner 2022-2023: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Banner_Summit/season_2022-2023_Banner_Summit.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_banner_2223.nc" \
    --out-dir "$OUT_BASE/banner_2223"

# Banner 2023-2024
echo "--- Banner 2023-2024: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Banner_Summit/season_2023-2024_Banner_Summit.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_banner_2324.nc" \
    --out-dir "$OUT_BASE/banner_2324"

# Banner 2024-2025
echo "--- Banner 2024-2025: $(date) ---"
conda run --no-capture-output -n sarvalanche python "$SCRIPT" \
    --nc "$NC_BASE/Banner_Summit/season_2024-2025_Banner_Summit.nc" \
    --hrrr "$HRRR_BASE/hrrr_temperature_banner_2425.nc" \
    --out-dir "$OUT_BASE/banner_2425"

echo "=== All done: $(date) ==="
