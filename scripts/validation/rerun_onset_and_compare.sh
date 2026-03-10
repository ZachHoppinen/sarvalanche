#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

NC_DIR="local/issw/netcdfs"
LOG="local/issw/rerun_onset.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# Re-run temporal onset on all CNN inference dirs
for inf_dir in "$NC_DIR"/*/v2_season_inference*; do
    [ -d "$inf_dir" ] || continue
    cnn_nc="$inf_dir/season_v2_debris_probabilities.nc"
    [ -f "$cnn_nc" ] || continue

    zone_dir="$(dirname "$inf_dir")"
    
    # Find the matching season_*.nc for this inference dir
    dir_name="$(basename "$inf_dir")"
    season=$(echo "$dir_name" | sed -E 's/^v2_season_inference_?//' | tr '_' '-')
    
    if [ -z "$season" ]; then
        # v2_season_inference (no season suffix) — find any season NC
        sar_nc=$(ls "$zone_dir"/season_*.nc 2>/dev/null | head -1)
    else
        # v2_season_inference_YYYY-YYYY — find matching NC
        sar_nc=$(ls "$zone_dir"/season_*${season}*.nc 2>/dev/null | head -1)
        if [ -z "$sar_nc" ]; then
            # Try without the dash
            season_underscore=$(echo "$season" | tr '-' '_')
            sar_nc=$(ls "$zone_dir"/season_*${season_underscore}*.nc 2>/dev/null | head -1)
        fi
        if [ -z "$sar_nc" ]; then
            sar_nc=$(ls "$zone_dir"/season_*.nc 2>/dev/null | head -1)
        fi
    fi
    
    if [ -z "$sar_nc" ] || [ ! -f "$sar_nc" ]; then
        log "SKIP $inf_dir — no SAR NC found"
        continue
    fi
    
    onset_nc="$inf_dir/temporal_onset.nc"
    log "=== Temporal onset: $(basename "$zone_dir")/$(basename "$inf_dir") ==="
    log "  CNN: $cnn_nc"
    log "  SAR: $sar_nc"
    
    conda run -n sarvalanche python -u scripts/debris_pixel_classifier/v2/temporal_onset.py \
        --cnn-nc "$cnn_nc" \
        --sar-nc "$sar_nc" \
        --threshold 0.5 \
        --out "$onset_nc" \
        2>&1 | tee -a "$LOG" || log "FAILED: $inf_dir"
done

# Re-run comparison
log "=== Re-running comparison ==="
conda run -n sarvalanche python -u scripts/validation/compare_cnn_to_observations.py \
    --max-time-gap 12 \
    2>&1 | tee -a "$LOG" || log "Comparison FAILED"

log "=== DONE ==="
