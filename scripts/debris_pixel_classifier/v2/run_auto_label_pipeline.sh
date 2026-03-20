#!/bin/bash
# Run auto-labeling + patch extraction for all SNFAC zones and seasons.
# Then run the training experiments.
#
# Usage:
#   conda run -n sarvalanche bash scripts/debris_pixel_classifier/v2/run_auto_label_pipeline.sh

set -e

FOOTPRINTS="local/debris_shapes/SNFAC"
AUTO_LABELS="local/debris_shapes/SNFAC_auto"
AUTO_PATCHES="local/experiments/auto_patches"
EXPERIMENT_DIR="local/experiments/auto_vs_human"

# ── Zone/season → NC mapping + dates ────────────────────────────────
# Sawtooth 2021-2022
echo "=== Auto-labeling Sawtooth 2021-2022 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/sawtooth_2021_2022" \
    --dates 2021-12-26

# Sawtooth 2022-2023
echo "=== Auto-labeling Sawtooth 2022-2023 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/sawtooth_2022_2023" \
    --dates 2022-12-01 2023-03-10

# Sawtooth 2023-2024
echo "=== Auto-labeling Sawtooth 2023-2024 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/sawtooth_2023_2024" \
    --dates 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29

# Sawtooth 2024-2025
echo "=== Auto-labeling Sawtooth 2024-2025 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/sawtooth_2024_2025" \
    --dates 2024-11-15 2024-12-29 2025-02-04 2025-02-19 2025-03-15 2025-04-10

# Banner 2022-2023
echo "=== Auto-labeling Banner 2022-2023 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Banner_Summit/season_2022-2023_Banner_Summit.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/banner_2022_2023" \
    --dates 2022-12-01 2023-03-10

# Banner 2023-2024
echo "=== Auto-labeling Banner 2023-2024 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Banner_Summit/season_2023-2024_Banner_Summit.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/banner_2023_2024" \
    --dates 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29

# Galena 2022-2023
echo "=== Auto-labeling Galena 2022-2023 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/galena_2022_2023" \
    --dates 2022-12-01 2023-03-10

# Galena 2023-2024
echo "=== Auto-labeling Galena 2023-2024 ==="
python scripts/debris_pixel_classifier/v2/auto_label.py \
    --nc "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc" \
    --footprints-dir "$FOOTPRINTS" \
    --out-dir "$AUTO_LABELS/galena_2023_2024" \
    --dates 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29

echo ""
echo "=== Auto-labeling complete ==="
echo ""

# ── Extract patches from auto labels ────────────────────────────────
# For each zone/season, extract patches using the same NC and auto-labeled gpkgs

extract_patches() {
    local NC="$1"
    local LABELS_DIR="$2"
    local PATCHES_OUT="$3"
    local DATE="$4"

    local GPKG="$LABELS_DIR/avalanche_labels_${DATE}.gpkg"
    local GEOTIFF_DIR="$LABELS_DIR/geotiffs/${DATE}"

    if [ ! -f "$GPKG" ]; then
        echo "  Skipping $DATE (no gpkg at $GPKG)"
        return
    fi

    echo "  Extracting patches for $DATE..."
    local OUT_DIR="$PATCHES_OUT/$DATE"

    local GEOTIFF_ARG=""
    if [ -d "$GEOTIFF_DIR" ] && [ "$(ls -A "$GEOTIFF_DIR"/*.tif 2>/dev/null)" ]; then
        GEOTIFF_ARG="--geotiff-dir $GEOTIFF_DIR"
    fi

    python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
        --nc "$NC" \
        --polygons "$GPKG" \
        $GEOTIFF_ARG \
        --date "$DATE" \
        --tau 6 \
        --out-dir "$OUT_DIR" \
        --stride 64 \
        --neg-ratio 3.0 || echo "  WARNING: extraction failed for $DATE"
}

echo "=== Extracting patches from auto labels ==="

# Sawtooth 2021-2022
NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc"
for DATE in 2021-12-26; do
    extract_patches "$NC" "$AUTO_LABELS/sawtooth_2021_2022" "$AUTO_PATCHES/sawtooth_2021_2022" "$DATE"
done

# Sawtooth 2022-2023
NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc"
for DATE in 2022-12-01 2023-03-10; do
    extract_patches "$NC" "$AUTO_LABELS/sawtooth_2022_2023" "$AUTO_PATCHES/sawtooth_2022_2023" "$DATE"
done

# Sawtooth 2023-2024
NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc"
for DATE in 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29; do
    extract_patches "$NC" "$AUTO_LABELS/sawtooth_2023_2024" "$AUTO_PATCHES/sawtooth_2023_2024" "$DATE"
done

# Sawtooth 2024-2025
NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
for DATE in 2024-11-15 2024-12-29 2025-02-04 2025-02-19 2025-03-15 2025-04-10; do
    extract_patches "$NC" "$AUTO_LABELS/sawtooth_2024_2025" "$AUTO_PATCHES/sawtooth_2024_2025" "$DATE"
done

# Banner 2022-2023
NC="local/issw/netcdfs/Banner_Summit/season_2022-2023_Banner_Summit.nc"
for DATE in 2022-12-01 2023-03-10; do
    extract_patches "$NC" "$AUTO_LABELS/banner_2022_2023" "$AUTO_PATCHES/banner_2022_2023" "$DATE"
done

# Banner 2023-2024
NC="local/issw/netcdfs/Banner_Summit/season_2023-2024_Banner_Summit.nc"
for DATE in 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29; do
    extract_patches "$NC" "$AUTO_LABELS/banner_2023_2024" "$AUTO_PATCHES/banner_2023_2024" "$DATE"
done

# Galena 2022-2023
NC="local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc"
for DATE in 2022-12-01 2023-03-10; do
    extract_patches "$NC" "$AUTO_LABELS/galena_2022_2023" "$AUTO_PATCHES/galena_2022_2023" "$DATE"
done

# Galena 2023-2024
NC="local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc"
for DATE in 2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29; do
    extract_patches "$NC" "$AUTO_LABELS/galena_2023_2024" "$AUTO_PATCHES/galena_2023_2024" "$DATE"
done

echo ""
echo "=== Patch extraction complete ==="

# ── Assign confidence to auto patches ────────────────────────────────
echo "=== Assigning confidence scores ==="
python scripts/debris_pixel_classifier/v2/assign_confidence.py \
    --patches-dir "$AUTO_PATCHES"

echo ""
echo "=== Pipeline complete ==="
echo "Auto labels: $AUTO_LABELS"
echo "Auto patches: $AUTO_PATCHES"
echo ""
echo "To run experiments:"
echo "  conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/run_training_experiments.py \\"
echo "      --human-patches local/issw/v2_patches \\"
echo "      --auto-patches $AUTO_PATCHES \\"
echo "      --out-dir $EXPERIMENT_DIR \\"
echo "      --epochs 50 --batch-size 4"
