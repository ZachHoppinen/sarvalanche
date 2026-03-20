#!/bin/bash
# Combo experiment: pooled + per-pair SAR maps fed together.
# Re-extracts patches in combo mode, trains pretrain→finetune, runs inference.
set -e

HUMAN_PATCHES="local/experiments/combo_patches/human"
AUTO_PATCHES="local/experiments/combo_patches/auto"
EXP_DIR="local/experiments/combo_mode"

extract_combo() {
    local NC="$1"
    local LABELS_DIR="$2"
    local PATCHES_OUT="$3"
    local DATE="$4"

    local GPKG="$LABELS_DIR/avalanche_labels_${DATE}.gpkg"
    [ ! -f "$GPKG" ] && return

    local GEOTIFF_DIR="$LABELS_DIR/geotiffs/${DATE}"
    local GEOTIFF_ARG=""
    if [ -d "$GEOTIFF_DIR" ] && ls "$GEOTIFF_DIR"/*.tif >/dev/null 2>&1; then
        GEOTIFF_ARG="--geotiff-dir $GEOTIFF_DIR"
    fi

    python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
        --nc "$NC" \
        --polygons "$GPKG" \
        $GEOTIFF_ARG \
        --date "$DATE" \
        --tau 6 \
        --out-dir "$PATCHES_OUT/$DATE" \
        --stride 64 \
        --neg-ratio 3.0 \
        --combo \
        --max-pairs 4 || echo "  WARNING: failed $DATE"
}

echo "=== Extracting HUMAN patches in combo mode ==="
for SEASON_NC in \
    "2021-2022:local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2021-2022_Sawtooth_&_Western_Smoky_Mtns.nc:2021-12-26" \
    "2022-2023:local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2022-2023_Sawtooth_&_Western_Smoky_Mtns.nc:2022-12-01 2023-03-10" \
    "2023-2024:local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2023-2024_Sawtooth_&_Western_Smoky_Mtns.nc:2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29" \
    "2024-2025:local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc:2024-11-15 2024-12-29 2025-02-04 2025-02-19 2025-03-15 2025-04-10"
do
    SEASON=$(echo "$SEASON_NC" | cut -d: -f1)
    NC=$(echo "$SEASON_NC" | cut -d: -f2)
    DATES=$(echo "$SEASON_NC" | cut -d: -f3)
    echo "--- Sawtooth $SEASON ---"
    for D in $DATES; do
        extract_combo "$NC" "local/debris_shapes/SNFAC" "$HUMAN_PATCHES/sawtooth" "$D"
    done
done

for SEASON_NC in \
    "2022-2023:local/issw/netcdfs/Banner_Summit/season_2022-2023_Banner_Summit.nc:2022-12-01 2023-03-10" \
    "2023-2024:local/issw/netcdfs/Banner_Summit/season_2023-2024_Banner_Summit.nc:2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29"
do
    SEASON=$(echo "$SEASON_NC" | cut -d: -f1)
    NC=$(echo "$SEASON_NC" | cut -d: -f2)
    DATES=$(echo "$SEASON_NC" | cut -d: -f3)
    echo "--- Banner $SEASON ---"
    for D in $DATES; do
        extract_combo "$NC" "local/debris_shapes/SNFAC" "$HUMAN_PATCHES/banner_${SEASON//-/_}" "$D"
    done
done

for SEASON_NC in \
    "2022-2023:local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2022-2023_Galena_Summit_&_Eastern_Mtns.nc:2022-12-01 2023-03-10" \
    "2023-2024:local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns/season_2023-2024_Galena_Summit_&_Eastern_Mtns.nc:2024-01-10 2024-01-12 2024-02-04 2024-02-26 2024-02-29"
do
    SEASON=$(echo "$SEASON_NC" | cut -d: -f1)
    NC=$(echo "$SEASON_NC" | cut -d: -f2)
    DATES=$(echo "$SEASON_NC" | cut -d: -f3)
    echo "--- Galena $SEASON ---"
    for D in $DATES; do
        extract_combo "$NC" "local/debris_shapes/SNFAC" "$HUMAN_PATCHES/galena_${SEASON//-/_}" "$D"
    done
done

echo ""
echo "=== Extracting AUTO patches in combo mode ==="
for ZONE_DIR in local/debris_shapes/SNFAC_auto/*/ local/debris_shapes/SNFAC_auto_expanded/*/; do
    [ ! -d "$ZONE_DIR" ] && continue
    ZONE_NAME=$(basename "$ZONE_DIR")
    PARENT=$(basename "$(dirname "$ZONE_DIR")")
    case "$ZONE_NAME" in
        banner_*)   NC_BASE="local/issw/netcdfs/Banner_Summit" ;;
        galena_*)   NC_BASE="local/issw/netcdfs/Galena_Summit_&_Eastern_Mtns" ;;
        sawtooth_*) NC_BASE="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns" ;;
        *) continue ;;
    esac
    SEASON=$(echo "$ZONE_NAME" | grep -oE '[0-9]{4}_[0-9]{4}' | sed 's/_/-/')
    NC=$(ls "$NC_BASE"/season_*"$SEASON"*.nc "$NC_BASE"/season_*"$(echo $SEASON | tr '-' '_')"*.nc 2>/dev/null | head -1)
    [ -z "$NC" ] && continue

    PREFIX=""
    [ "$PARENT" = "SNFAC_auto_expanded" ] && PREFIX="expanded_"

    echo "--- ${PREFIX}${ZONE_NAME} ---"
    for GPKG in "$ZONE_DIR"/avalanche_labels_*.gpkg; do
        DATE=$(basename "$GPKG" | sed 's/avalanche_labels_//' | sed 's/.gpkg//')
        extract_combo "$NC" "$ZONE_DIR" "$AUTO_PATCHES/${PREFIX}${ZONE_NAME}" "$DATE"
    done
done

echo ""
echo "=== Assigning confidence ==="
python scripts/debris_pixel_classifier/v2/assign_confidence.py --patches-dir "$HUMAN_PATCHES"
python scripts/debris_pixel_classifier/v2/assign_confidence.py --patches-dir "$AUTO_PATCHES"

echo ""
echo "=== Training (pretrain→finetune only) ==="
python scripts/debris_pixel_classifier/v2/run_training_experiments.py \
    --human-patches "$HUMAN_PATCHES" \
    --auto-patches "$AUTO_PATCHES" \
    --out-dir "$EXP_DIR" \
    --epochs 50 --batch-size 4 \
    --finetune-epochs 20 --finetune-lr 1e-4 \
    --sar-channels 3

echo ""
echo "=== Season inference ==="
NC="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
OUTDIR="local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference_2024-2025_combo_pt_ft"
WEIGHTS="$EXP_DIR/exp3_pretrain_finetune/best.pt"

if [ -f "$WEIGHTS" ]; then
    python scripts/debris_pixel_classifier/v2/full_season_inference.py \
        --nc "$NC" \
        --weights "$WEIGHTS" \
        --season 2024-2025 \
        --tau 6 \
        --out-dir "$OUTDIR" \
        --no-tiffs \
        --stride 32 \
        --batch-size 16 \
        --combo \
        --max-pairs 4
else
    echo "WARNING: No weights at $WEIGHTS — skipping inference"
fi

echo ""
echo "=== Done ==="
