#!/usr/bin/env bash
# Full Chugach (CNFAIC) pipeline: build datasets, CNN inference, temporal onset,
# FlowPy on observations, and compare detections to observations.
#
# Usage:
#   bash scripts/validation/run_chugach_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

WEIGHTS="local/issw/v2_patches/v2_detector_best.pt"
OUT_DIR="local/cnfaic"
NC_DIR="$OUT_DIR/netcdfs"
OBS_DIR="$OUT_DIR/observations"
LOG="$OUT_DIR/pipeline.log"

mkdir -p "$NC_DIR" "$OBS_DIR"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# CNFAIC zones (all 4)
ZONES=(
    "Turnagain Pass and Girdwood"
    "Summit Lake"
    "Chugach State Park"
    "Seward and Lost Lake"
)

# Seasons with observations
SEASONS=("2024-2025" "2025-2026")

# ── 1. Build season datasets ─────────────────────────────────────────────
for zone in "${ZONES[@]}"; do
    safe_zone="${zone// /_}"
    safe_zone="${safe_zone////-}"
    zone_dir="$NC_DIR/$safe_zone"
    mkdir -p "$zone_dir"

    for season in "${SEASONS[@]}"; do
        nc_fname="season_${season}_${safe_zone}.nc"
        nc_path="$zone_dir/$nc_fname"

        if [ -f "$nc_path" ]; then
            log "SKIP dataset build: $zone $season — already exists"
            continue
        fi

        log "=== Building dataset: $zone $season ==="
        conda run -n sarvalanche python scripts/issw_analysis/build_season_dataset.py \
            --zone "$zone" \
            --center CNFAIC \
            --season "$season" \
            --out-dir "$NC_DIR" \
            --baseline-days 60 \
            2>&1 | tee -a "$LOG" || {
                log "FAILED dataset build: $zone $season"
                continue
            }
        log "Done dataset build: $zone $season"
    done
done

# ── 2. CNN inference + temporal onset ─────────────────────────────────────
for zone in "${ZONES[@]}"; do
    safe_zone="${zone// /_}"
    safe_zone="${safe_zone////-}"
    zone_dir="$NC_DIR/$safe_zone"

    for nc in "$zone_dir"/season_*.nc; do
        [ -f "$nc" ] || continue

        # Extract season from filename
        season=$(echo "$(basename "$nc")" | sed -E 's/^season_([0-9]{4}-[0-9]{4})_.*/\1/')

        inf_dir="$zone_dir/v2_season_inference_${season}"
        cnn_nc="$inf_dir/season_v2_debris_probabilities.nc"
        onset_nc="$inf_dir/temporal_onset.nc"

        # CNN inference
        if [ -f "$cnn_nc" ]; then
            log "SKIP CNN inference: $zone $season — already exists"
        else
            log "=== CNN inference: $zone $season ==="
            mkdir -p "$inf_dir"
            if conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/full_season_inference.py \
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
                log "Done CNN inference: $zone $season"
            else
                log "FAILED CNN inference: $zone $season"
                continue
            fi
        fi

        # Temporal onset
        if [ -f "$onset_nc" ]; then
            log "SKIP temporal onset: $zone $season — already exists"
        elif [ -f "$cnn_nc" ]; then
            log "=== Temporal onset: $zone $season ==="
            if conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/temporal_onset.py \
                --cnn-nc "$cnn_nc" \
                --sar-nc "$nc" \
                --threshold 0.5 \
                --out "$onset_nc" \
                2>&1 | tee -a "$LOG"; then
                log "Done temporal onset: $zone $season"
            else
                log "FAILED temporal onset: $zone $season"
            fi
        fi
    done
done

# ── 3. FlowPy on observations ────────────────────────────────────────────
log "=== Running FlowPy on CNFAIC observations ==="
conda run -n sarvalanche python scripts/validation/run_flowpy_on_observations_generic.py \
    --obs-csv "$OUT_DIR/cnfaic_obs_all.csv" \
    --nc-dir "$NC_DIR" \
    --out-dir "$OBS_DIR" \
    2>&1 | tee -a "$LOG" || log "FlowPy observations had errors"

# ── 4. Compare detections to observations ─────────────────────────────────
log "=== Comparing CNN detections to observations ==="
conda run -n sarvalanche python scripts/validation/compare_observations_generic.py \
    --obs-csv "$OUT_DIR/cnfaic_obs_all.csv" \
    --paths-gpkg "$OBS_DIR/all_flowpy_paths.gpkg" \
    --nc-dir "$NC_DIR" \
    --out-csv "$OBS_DIR/comparison_summary.csv" \
    --max-time-gap 12 \
    2>&1 | tee -a "$LOG" || log "Comparison had errors"

log "=== CHUGACH PIPELINE COMPLETE ==="
log "Results at: $OBS_DIR/comparison_summary.csv"
