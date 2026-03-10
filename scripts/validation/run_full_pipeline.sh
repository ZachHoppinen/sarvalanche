#!/usr/bin/env bash
# Full automation: wait for builds, run CNN inference + temporal onset on all
# new season datasets, re-run FlowPy observations, re-run comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

WEIGHTS="local/issw/v2_patches/v2_detector_best.pt"
NC_DIR="local/issw/netcdfs"
LOG="local/issw/pipeline_automation.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# ── 0. Wait for build_season_dataset chain to finish ──────────────────────
BUILD_PID=14742
if kill -0 "$BUILD_PID" 2>/dev/null; then
    log "Waiting for build chain (PID $BUILD_PID) to finish..."
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        sleep 120
    done
    log "Build chain finished."
else
    log "Build chain already finished."
fi

log "Available season NCs:"
ls -1 "$NC_DIR"/*/season_*.nc 2>/dev/null | tee -a "$LOG"

# ── 1. Run CNN inference + temporal onset on each season NC ───────────────
for nc in "$NC_DIR"/*/season_*_*.nc; do
    [ -f "$nc" ] || continue
    zone_dir="$(dirname "$nc")"
    zone_name="$(basename "$zone_dir")"
    nc_base="$(basename "$nc")"

    # Extract season from filename: season_YYYY-YYYY_Zone.nc or season_YYYY_YYYY_Zone.nc
    season=$(echo "$nc_base" | sed -E 's/^season_([0-9]{4}[-_][0-9]{4})_.*/\1/' | tr '_' '-')

    # Per-season inference directory
    inf_dir="$zone_dir/v2_season_inference_${season}"
    cnn_nc="$inf_dir/season_v2_debris_probabilities.nc"
    onset_nc="$inf_dir/temporal_onset.nc"

    # Also check the original v2_season_inference dir (for pre-existing inference)
    orig_cnn="$zone_dir/v2_season_inference/season_v2_debris_probabilities.nc"
    orig_onset="$zone_dir/v2_season_inference/temporal_onset.nc"

    # Skip if per-season inference already done
    if [ -f "$cnn_nc" ] && [ -f "$onset_nc" ]; then
        log "SKIP $zone_name $season — per-season inference exists"
        continue
    fi

    # Skip if original inference exists AND there's only one season NC for this zone
    # (i.e., the original inference covers this NC)
    n_ncs=$(ls "$zone_dir"/season_*.nc 2>/dev/null | wc -l | tr -d ' ')
    if [ -f "$orig_cnn" ] && [ -f "$orig_onset" ] && [ "$n_ncs" -eq 1 ]; then
        log "SKIP $zone_name $season — original inference covers single NC"
        continue
    fi

    log "=== CNN inference: $zone_name $season ==="
    mkdir -p "$inf_dir"

    # Step 1: Full season inference
    if [ ! -f "$cnn_nc" ]; then
        log "  Running full_season_inference.py..."
        if conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/full_season_inference.py \
            --nc "$nc" \
            --weights "$WEIGHTS" \
            --season "$season" \
            --tau 6 \
            --out-dir "$inf_dir" \
            --no-tiffs \
            --stride 32 \
            --batch-size 16 \
            2>&1 | tee -a "$LOG"; then
            log "  CNN inference done."
        else
            log "  CNN inference FAILED for $zone_name $season — skipping"
            continue
        fi
    fi

    # Step 2: Temporal onset
    if [ -f "$cnn_nc" ] && [ ! -f "$onset_nc" ]; then
        log "  Running temporal_onset.py..."
        if conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/temporal_onset.py \
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

# ── 2. Re-run FlowPy on observations (auto-discovers all NCs) ────────────
log "=== Re-running FlowPy on observations ==="
conda run -n sarvalanche python scripts/validation/run_flowpy_on_observations.py \
    2>&1 | tee -a "$LOG" || log "FlowPy observations had errors"

# ── 3. Re-run comparison (auto-discovers all CNN sources) ─────────────────
log "=== Re-running CNN comparison ==="
conda run -n sarvalanche python scripts/validation/compare_cnn_to_observations.py \
    --max-time-gap 12 \
    2>&1 | tee -a "$LOG" || log "Comparison had errors"

log "=== PIPELINE COMPLETE ==="
log "Results at: local/issw/observations/comparison_summary.csv"
