# Evaluation Pipeline: CNN Detections vs Avalanche Observations

## Overview

Two-step pipeline to evaluate pairwise debris CNN detections against reported avalanche observations across US forecast centers.

**Step 1** runs CNN inference on season datasets and saves pair probabilities (expensive, run once per model).
**Step 2** runs temporal onset and matches detections against observations (cheap, re-runnable with different parameters).

## Step 1: Run Inference

Runs pairwise CNN on all SAR pairs in a season dataset. Saves per-pair probability maps as `.npz` and metadata as `.csv`.

```bash
# Single zone
conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
    --nc local/issw/uac/netcdfs/Salt_Lake/season_2023-2024_Salt_Lake.nc \
    --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v3_best.pt \
    --out-dir local/issw/uac/inference/Salt_Lake/

# All zones for a center
conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
    --nc-dir local/issw/uac/netcdfs/ \
    --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v3_best.pt \
    --out-dir local/issw/uac/inference/

# All centers (batch)
for center in uac snfac caic fac btac esac cnfaic nwac; do
    conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
        --nc-dir local/issw/$center/netcdfs/ \
        --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v3_best.pt \
        --out-dir local/issw/$center/inference/
done
```

### Options
- `--weights` — model checkpoint. v2 (9ch) and v3 (11ch with post_vv/post_vh) auto-detected
- `--stride 32` — sliding window stride in pixels (lower = more overlap = slower but better)
- `--batch-size 16` — inference batch size
- `--max-span-days 60` — max pair span
- `--device cpu|mps|cuda` — torch device (auto-detected if omitted)

### Output
```
inference/{Zone_Name}/
├── season_2023-2024_{Zone}_pair_probs.npz   # compressed probability maps
├── season_2023-2024_{Zone}_pair_meta.csv    # pair metadata (track, t_start, t_end)
├── season_2024-2025_{Zone}_pair_probs.npz
└── season_2024-2025_{Zone}_pair_meta.csv
```

Skips zones that already have inference results (delete `.npz` to re-run).

## Step 2: Evaluate Against Observations

Loads saved inference, runs temporal onset, matches detections against observation coordinates using FlowPy paths.

```bash
# UAC (uses UAC's own observation format)
conda run -n sarvalanche python scripts/evaluation/evaluate_observations.py \
    --obs local/issw/uac/uac_avalanche_observations.csv \
    --inference-dir local/issw/uac/inference/ \
    --nc-dir local/issw/uac/netcdfs/ \
    --out local/issw/uac/eval_v3.csv

# SNFAC (uses avalanche.org observation format)
conda run -n sarvalanche python scripts/evaluation/evaluate_observations.py \
    --obs local/issw/snfac/snfac_avalanche_observations.csv \
    --inference-dir local/issw/snfac/inference/ \
    --nc-dir local/issw/snfac/netcdfs/ \
    --out local/issw/snfac/eval_v3.csv
```

### Options
- `--onset-threshold 0.2` — probability threshold for temporal onset firing
- `--onset-gap-days 18` — temporal gap to split distinct events
- `--min-d-size 2.0` — minimum D-size filter for observations

### Detection Logic
For each D2+ observation:
1. Project lat/lon to dataset CRS
2. Get FlowPy `cell_counts` within 500m of observation point
3. Run temporal onset on saved pair probabilities (cached after first run)
4. Check for connected cluster of 5+ detection pixels within the FlowPy path mask
5. Report detection at ±1, 3, 7, 10 day tolerances

### Output
CSV with columns: `date, lat, lon, region, zone, y_idx, x_idx, path_pixels, detected_1d, detected_3d, detected_7d, detected_10d`

Summary printed at end:
```
=== DETECTION SUMMARY ===
  ± 1d:  342/1847 detected (18.5%)
  ± 3d:  567/1847 detected (30.7%)
  ± 7d:  891/1847 detected (48.2%)
  ±10d: 1023/1847 detected (55.4%)
```

## Swapping Models

Run inference with a different model checkpoint — results go to separate dirs:

```bash
# v2 model
conda run -n sarvalanche python scripts/evaluation/run_zone_inference.py \
    --nc-dir local/issw/uac/netcdfs/ \
    --weights src/sarvalanche/ml/weights/pairwise_debris_detector/v2_best.pt \
    --out-dir local/issw/uac/inference_v2/

# Evaluate v2
conda run -n sarvalanche python scripts/evaluation/evaluate_observations.py \
    --obs local/issw/uac/uac_avalanche_observations.csv \
    --inference-dir local/issw/uac/inference_v2/ \
    --nc-dir local/issw/uac/netcdfs/ \
    --out local/issw/uac/eval_v2.csv
```

## Iterating on Temporal Onset

Re-run evaluation with different parameters without re-running inference:

```bash
# Delete cached onset to force re-computation
rm local/issw/uac/inference/*_onset.nc local/issw/uac/inference/*_onset.npz

# Re-evaluate with different threshold
conda run -n sarvalanche python scripts/evaluation/evaluate_observations.py \
    --obs local/issw/uac/uac_avalanche_observations.csv \
    --inference-dir local/issw/uac/inference/ \
    --nc-dir local/issw/uac/netcdfs/ \
    --out local/issw/uac/eval_v3_thresh015.csv \
    --onset-threshold 0.15
```

## Available Data

| Center | Zones | Seasons | D2+ Obs | Obs Source |
|--------|-------|---------|---------|------------|
| UAC | Salt Lake, Logan, Uintas, Provo, Skyline, Ogden | 23-24, 24-25 | 1,847 | `uac_avalanche_observations.csv` |
| SNFAC | Galena, Sawtooth, Soldier | 23-24, 24-25 | 1,097 | `snfac_avalanche_observations.csv` |
| CAIC | zone 2752 | 23-24, 24-25 | 917 | `caic_avalanche_observations.csv` |
| FAC | Flathead & Glacier NP | 23-24, 24-25 | 518 | fetch via `get_avalanche_observations.py` |
| BTAC | Tetons | 23-24, 24-25 | 487 | fetch via `get_avalanche_observations.py` |
| NWAC | Snoqualmie Pass | 23-24, 24-25 | 45 | fetch via `get_avalanche_observations.py` |
| ESAC | Eastside Region | 23-24, 24-25 | 173 | fetch via `get_avalanche_observations.py` |
| CNFAIC | Turnagain Pass | 23-24, 24-25 | 69 | fetch via `get_avalanche_observations.py` |
