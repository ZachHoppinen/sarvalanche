#!/bin/bash
# Expanded auto-labeling: 61 dates selected by SNFAC danger rating
# with 12-day minimum spacing to avoid overlapping avalanche cycles.
# Uses --d-threshold 4 for full-scene mode (no footprint windows).
#
# Selection: ~2 High/Extreme + 2 Considerable + 1 Moderate + 1 Low per zone-season
#
# Usage:
#   conda run -n sarvalanche bash scripts/debris_pixel_classifier/v2/run_expanded_auto_label.sh

set -e

AUTO_LABELS="local/debris_shapes/SNFAC_auto_expanded"
AUTO_PATCHES="local/experiments/auto_patches_expanded"
D_THRESH=4

auto_label() {
    local NC="$1"
    local OUT="$2"
    shift 2
    local DATES="$@"

    python scripts/debris_pixel_classifier/v2/auto_label.py \
        --nc "$NC" \
        --footprints-dir "local/debris_shapes/SNFAC" \
        --out-dir "$OUT" \
        --dates $DATES \
        --d-threshold $D_THRESH || echo "WARNING: auto_label failed for $OUT"
}

extract() {
    local NC="$1"
    local LABELS_DIR="$2"
    local PATCHES_OUT="$3"
    local DATE="$4"

    local GPKG="$LABELS_DIR/avalanche_labels_${DATE}.gpkg"
    if [ ! -f "$GPKG" ]; then
        return
    fi

    echo "  Extracting $DATE..."
    python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
        --nc "$NC" \
        --polygons "$GPKG" \
        --date "$DATE" \
        --tau 6 \
        --out-dir "$PATCHES_OUT/$DATE" \
        --stride 64 \
        --neg-ratio 3.0 || echo "  WARNING: extraction failed for $DATE"
}

# ── Phase 1: Auto-label ──────────────────────────────────────────────
echo "=== PHASE 1: Auto-labeling 61 dates ==="

echo "=== Banner 2022-2023 ==="
auto_label "local/issw/netcdfs/Banner_Summit/season_2022-2023_Banner_Summit.nc" \
    "$AUTO_LABELS/banner_2022_2023" \
    2022-12-03 2022-12-15 2022-12-27 2023-01-08 2023-01-20 2023-02-15

echo "=== Banner 2023-2024 ==="
auto_label "local/issw/netcdfs/Banner_Summit/season_2023-2024_Banner_Summit.nc" \
    "$AUTO_LABELS/banner_2023_2024" \
    2023-12-05 2023-12-17 2023-12-29 2024-01-17 2024-02-08

echo "=== Banner 2024-2025 ==="
auto_label "local/issw/netcdfs/Banner_Summit/season_2024-2025_Banner_Summit.nc" \
    "$AUTO_LABELS/banner_2024_2025" \
    2024-12-16 2024-12-28 2025-01-09 2025-01-21 2025-02-02 2025-02-14

echo "=== Galena 2021-2022 ==="
auto_label "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2021-2022_Galena_Summit_&_Eastern_Mtns.nc" \
    "$AUTO_LABELS/galena_2021_2022" \
    2021-12-11 2021-12-23 2022-01-13 2022-01-27 2022-03-26

echo "=== Galena 2022-2023 ==="
auto_label "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc" \
    "$AUTO_LABELS/galena_2022_2023" \
    2022-12-03 2022-12-15 2022-12-27 2023-01-08 2023-01-20 2023-04-14

echo "=== Galena 2023-2024 ==="
auto_label "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc" \
    "$AUTO_LABELS/galena_2023_2024" \
    2023-12-05 2023-12-17 2023-12-29 2024-01-17 2024-02-08

echo "=== Galena 2024-2025 ==="
auto_label "local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2024-2025_Galena_Summit_&_Eastern_Mtns.nc" \
    "$AUTO_LABELS/galena_2024_2025" \
    2024-12-16 2024-12-28 2025-01-09 2025-01-21 2025-02-02 2025-02-14

echo "=== Sawtooth 2021-2022 ==="
auto_label "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc" \
    "$AUTO_LABELS/sawtooth_2021_2022" \
    2021-12-11 2021-12-23 2022-01-13 2022-01-27 2022-03-26

echo "=== Sawtooth 2022-2023 ==="
auto_label "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc" \
    "$AUTO_LABELS/sawtooth_2022_2023" \
    2022-12-03 2022-12-15 2022-12-27 2023-01-08 2023-01-20 2023-02-15

echo "=== Sawtooth 2023-2024 ==="
auto_label "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc" \
    "$AUTO_LABELS/sawtooth_2023_2024" \
    2023-12-05 2023-12-17 2023-12-29 2024-01-17 2024-02-08

echo "=== Sawtooth 2024-2025 ==="
auto_label "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
    "$AUTO_LABELS/sawtooth_2024_2025" \
    2024-12-16 2024-12-28 2025-01-09 2025-01-21 2025-02-02 2025-02-14

echo ""
echo "=== PHASE 2: Extract patches ==="

# For each zone-season, extract patches for all auto-labeled dates
for ZONE_DIR in "$AUTO_LABELS"/*/; do
    ZONE_NAME=$(basename "$ZONE_DIR")

    # Determine NC path from zone name
    case "$ZONE_NAME" in
        banner_*)   NC_BASE="local/issw/netcdfs/Banner_Summit" ;;
        galena_*)   NC_BASE="local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns" ;;
        sawtooth_*) NC_BASE="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns" ;;
        *) echo "Unknown zone: $ZONE_NAME"; continue ;;
    esac

    # Extract season from dir name (e.g., banner_2022_2023 -> 2022-2023)
    SEASON=$(echo "$ZONE_NAME" | grep -oE '[0-9]{4}_[0-9]{4}' | sed 's/_/-/')

    # Find the NC file
    NC=$(ls "$NC_BASE"/season_*"$SEASON"*.nc "$NC_BASE"/season_*"$(echo $SEASON | tr '-' '_')"*.nc 2>/dev/null | head -1)
    if [ -z "$NC" ]; then
        echo "WARNING: No NC found for $ZONE_NAME"
        continue
    fi

    echo "--- $ZONE_NAME ($NC) ---"

    for GPKG in "$ZONE_DIR"/avalanche_labels_*.gpkg; do
        DATE=$(basename "$GPKG" | sed 's/avalanche_labels_//' | sed 's/.gpkg//')
        extract "$NC" "$ZONE_DIR" "$AUTO_PATCHES/$ZONE_NAME" "$DATE"
    done
done

echo ""
echo "=== PHASE 3: Assign confidence ==="
python scripts/debris_pixel_classifier/v2/assign_confidence.py \
    --patches-dir "$AUTO_PATCHES"

echo ""
echo "=== Expanded pipeline complete ==="
echo "Auto labels: $AUTO_LABELS"
echo "Auto patches: $AUTO_PATCHES"
