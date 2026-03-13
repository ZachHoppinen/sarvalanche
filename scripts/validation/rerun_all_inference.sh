#!/usr/bin/env bash
# Re-run CNN inference + temporal onset + comparison with the new skip+dice model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

WEIGHTS="local/issw/v2_patches/v2_detector_best.pt"
NC_DIR="local/issw/netcdfs"
LOG="local/issw/rerun_inference.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

log "=== Re-running all inference with skip+dice model ==="

# ── 1. CNN inference + temporal onset on each season NC ───────────────
for nc in "$NC_DIR"/*/season_*_*.nc; do
    [ -f "$nc" ] || continue
    zone_dir="$(dirname "$nc")"
    zone_name="$(basename "$zone_dir")"
    nc_base="$(basename "$nc")"

    # Extract season from filename
    season=$(echo "$nc_base" | sed -E 's/^season_([0-9]{4}[-_][0-9]{4})_.*/\1/' | tr '_' '-')

    inf_dir="$zone_dir/v2_season_inference_${season}"
    cnn_nc="$inf_dir/season_v2_debris_probabilities.nc"
    onset_nc="$inf_dir/temporal_onset.nc"

    mkdir -p "$inf_dir"

    # Step 1: Full season inference (skip if already done)
    if [ ! -f "$cnn_nc" ]; then
        log "=== CNN inference: $zone_name $season ==="
        if conda run -n sarvalanche python -u scripts/debris_pixel_classifier/v2/full_season_inference.py \
            --nc "$nc" \
            --weights "$WEIGHTS" \
            --season "$season" \
            --tau 6 \
            --out-dir "$inf_dir" \
            --no-tiffs \
            --stride 32 \
            --batch-size 16 \
            --skip \
            2>&1 | tee -a "$LOG"; then
            log "  CNN inference done."
        else
            log "  CNN inference FAILED for $zone_name $season — skipping"
            continue
        fi
    else
        log "SKIP inference $zone_name $season — already exists"
    fi

    # Step 2: Temporal onset (skip if already done)
    if [ -f "$cnn_nc" ] && [ ! -f "$onset_nc" ]; then
        log "  Running temporal_onset.py..."
        if conda run -n sarvalanche python -u scripts/debris_pixel_classifier/v2/temporal_onset.py \
            --cnn-nc "$cnn_nc" \
            --sar-nc "$nc" \
            --threshold 0.5 \
            --out "$onset_nc" \
            2>&1 | tee -a "$LOG"; then
            log "  Temporal onset done."
        else
            log "  Temporal onset FAILED for $zone_name $season"
        fi
    fi
done

# ── 2. Re-run comparison ─────────────────────────────────────────────
log "=== Re-running CNN comparison ==="
conda run -n sarvalanche python -u scripts/validation/compare_cnn_to_observations.py \
    --max-time-gap 12 \
    2>&1 | tee -a "$LOG" || log "Comparison had errors"

log "=== PIPELINE COMPLETE ==="
log "Results at: local/issw/observations/comparison_summary.csv"
