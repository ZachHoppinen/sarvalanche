#!/bin/zsh
# Extract per-pair v2.1 patches: individual crossing pairs instead of pooled change.
# SAR channels: [log1p_change, ANF, temporal_proximity]
set -e

SCRIPT="scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py"
SHAPES="local/issw/debris_shapes"
OUT="local/issw/v2_patches_pairs"

# ── Sawtooth zone ──────────────────────────────────────────────────
SAW="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns"
typeset -A SAW_DATES
SAW_DATES=(
  2021-12-26 "season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc"
  2023-03-10 "season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-11-15 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2024-12-29 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-02-04 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-02-19 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-03-15 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
  2025-04-10 "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
)

for DATE NC in "${(@kv)SAW_DATES}"; do
  if [[ -f "${OUT}/${DATE}/labels.json" ]]; then
    echo "=== Skipping Sawtooth ${DATE} (already extracted) ==="
    continue
  fi
  echo ""
  echo "=== Extracting Sawtooth ${DATE} ==="
  conda run -n sarvalanche python "${SCRIPT}" \
    --nc "${SAW}/${NC}" \
    --polygons "${SHAPES}/avalanche_labels_${DATE}.gpkg" \
    --geotiff-dir "${SHAPES}/geotiffs/${DATE}" \
    --date "${DATE}" --tau 6 \
    --out-dir "${OUT}/${DATE}" --stride 64 --neg-ratio 3.0 \
    --pairs --max-pairs 4
done

# ── Banner zone ──────────────────────────────────────────────────
BAN="local/issw/netcdfs/Banner_Summit"
typeset -A BAN_DATES
BAN_DATES=(
  2023-03-10 "season_2022-2023_Banner_Summit.nc"
  2024-01-10 "season_2023-2024_Banner_Summit.nc"
)

for DATE NC in "${(@kv)BAN_DATES}"; do
  if [[ "${NC}" == *"2022-2023"* ]]; then SUBDIR="banner_2022_2023"; else SUBDIR="banner_2023_2024"; fi
  if [[ -f "${OUT}/${SUBDIR}/${DATE}/labels.json" ]]; then
    echo "=== Skipping Banner ${DATE} (already extracted) ==="
    continue
  fi
  echo ""
  echo "=== Extracting Banner ${DATE} ==="
  conda run -n sarvalanche python "${SCRIPT}" \
    --nc "${BAN}/${NC}" \
    --polygons "${SHAPES}/avalanche_labels_${DATE}.gpkg" \
    --geotiff-dir "${SHAPES}/geotiffs/${DATE}" \
    --date "${DATE}" --tau 6 \
    --out-dir "${OUT}/${SUBDIR}/${DATE}" --stride 64 --neg-ratio 3.0 \
    --pairs --max-pairs 4
done

# ── Galena zone ──────────────────────────────────────────────────
GAL="local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns"
typeset -A GAL_DATES
GAL_DATES=(
  2022-12-01 "season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc"
  2024-01-12 "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc"
  2024-02-04 "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc"
  2024-02-26 "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc"
  2024-02-29 "season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc"
)

for DATE NC in "${(@kv)GAL_DATES}"; do
  if [[ "${NC}" == *"2022-2023"* ]]; then SUBDIR="galena_2022_2023"; else SUBDIR="galena_2023_2024"; fi
  if [[ -f "${OUT}/${SUBDIR}/${DATE}/labels.json" ]]; then
    echo "=== Skipping Galena ${DATE} (already extracted) ==="
    continue
  fi
  echo ""
  echo "=== Extracting Galena ${DATE} ==="
  conda run -n sarvalanche python "${SCRIPT}" \
    --nc "${GAL}/${NC}" \
    --polygons "${SHAPES}/avalanche_labels_${DATE}.gpkg" \
    --geotiff-dir "${SHAPES}/geotiffs/${DATE}" \
    --date "${DATE}" --tau 6 \
    --out-dir "${OUT}/${SUBDIR}/${DATE}" --stride 64 --neg-ratio 3.0 \
    --pairs --max-pairs 4
done

echo ""
echo "=== All done ==="
echo "Patches in: ${OUT}"
