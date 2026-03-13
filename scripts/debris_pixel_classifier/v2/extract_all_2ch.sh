#!/bin/zsh
# Extract 2-channel v2 patches for all 14 labeled dates.
set -e

# Skip dates that are known to have no overlap with their season dataset
SKIP_DATES="2022-12-01"

BASE="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns"
SHAPES="local/issw/debris_shapes"
OUT="local/issw/v2_patches_2ch"
SCRIPT="scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py"

typeset -A DATE_TO_SEASON
DATE_TO_SEASON=(
  2021-12-26 "season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc"
  2022-12-01 "season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc"
  2023-03-10 "season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-01-10 "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-01-12 "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-02-04 "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-02-26 "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-02-29 "season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-11-15 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-12-29 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-02-04 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-02-19 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-03-15 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-04-10 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
)

DATES=(
  2021-12-26 2022-12-01 2023-03-10
  2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29
  2024-11-15 2024-12-29 2025-02-04 2025-02-19 2025-03-15 2025-04-10
)

for DATE in "${DATES[@]}"; do
  NC="${BASE}/${DATE_TO_SEASON[$DATE]}"
  POLY="${SHAPES}/avalanche_labels_${DATE}.gpkg"
  GEOTIFFS="${SHAPES}/geotiffs/${DATE}"

  # Skip known-bad dates
  if [[ "$SKIP_DATES" == *"$DATE"* ]]; then
    echo "=== Skipping ${DATE} (no dataset overlap) ==="
    continue
  fi

  # Skip already-extracted dates
  if [[ -f "${OUT}/${DATE}/labels.json" ]]; then
    echo "=== Skipping ${DATE} (already extracted) ==="
    continue
  fi

  echo ""
  echo "=== Extracting ${DATE} ==="

  conda run -n sarvalanche python "${SCRIPT}" \
    --nc "${NC}" \
    --polygons "${POLY}" \
    --geotiff-dir "${GEOTIFFS}" \
    --date "${DATE}" \
    --tau 6 \
    --out-dir "${OUT}/${DATE}" \
    --stride 64 \
    --neg-ratio 3.0
done

echo ""
echo "=== All done ==="
