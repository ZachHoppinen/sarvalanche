#!/bin/bash
set -e

NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
LABELS_DIR="local/debris_shapes/SNFAC"
VAL_PATHS="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/snfac_obs_flowpy_paths_2024_2025.gpkg"
OUT_BASE="local/issw/v3_experiment/patches"

rm -rf "$OUT_BASE"/*/

echo "=== Extracting all dates (single invocation) ==="
conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/extract_patches.py \
    --nc "$NC" \
    --date 2024-11-15 2024-12-29 2025-02-04 2025-02-19 2025-03-15 2025-04-10 \
    --polygons \
        "$LABELS_DIR/avalanche_labels_2024-11-15.gpkg" \
        "$LABELS_DIR/avalanche_labels_2024-12-29.gpkg" \
        "$LABELS_DIR/avalanche_labels_2025-02-04.gpkg" \
        "$LABELS_DIR/avalanche_labels_2025-02-19.gpkg" \
        "$LABELS_DIR/avalanche_labels_2025-03-15.gpkg" \
        "$LABELS_DIR/avalanche_labels_2025-04-10.gpkg" \
    --out-dir \
        "$OUT_BASE/2024-11-15" "$OUT_BASE/2024-12-29" "$OUT_BASE/2025-02-04" \
        "$OUT_BASE/2025-02-19" "$OUT_BASE/2025-03-15" "$OUT_BASE/2025-04-10" \
    --geotiff-dir \
        "$LABELS_DIR/geotiffs/2024-11-15" "$LABELS_DIR/geotiffs/2024-12-29" \
        "$LABELS_DIR/geotiffs/2025-02-04" "$LABELS_DIR/geotiffs/2025-02-19" \
        "$LABELS_DIR/geotiffs/2025-03-15" "$LABELS_DIR/geotiffs/2025-04-10" \
    --val-paths "$VAL_PATHS" \
    --stride 64 --neg-ratio 3.0

echo "=== EXTRACTION DONE ==="

conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/train.py \
    --data-dir "$OUT_BASE/2024-11-15" "$OUT_BASE/2024-12-29" "$OUT_BASE/2025-02-04" \
               "$OUT_BASE/2025-02-19" "$OUT_BASE/2025-03-15" "$OUT_BASE/2025-04-10" \
    --epochs 50 --lr 1e-3 --base-ch 16 \
    --out local/issw/v3_experiment/v3_snfac_best.pt

echo "=== TRAINING DONE ==="

conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v3/evaluate_vs_snfac.py \
    --weights local/issw/v3_experiment/v3_snfac_best.pt \
    --nc "$NC" \
    --hrrr local/issw/hrrr_temperature_sawtooth_2425.nc \
    --zone Sawtooth --season 2024-2025 --stride 32 --batch-size 16 \
    --max-span-days 30
