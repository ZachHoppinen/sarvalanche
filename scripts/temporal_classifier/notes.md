# Temporal Classifier (Stage 2)

Stage 2 of the two-stage avalanche detection pipeline.

- **Stage 1** (CNN, in `scripts/debris_pixel_classifier/v2/`): Per-acquisition-date sliding-window CNN produces debris probability maps. Output: `season_v2_debris_probabilities.nc` with `debris_probability(time, y, x)`.
- **Stage 2** (this directory): Analyzes the probability time series to determine *when* avalanches occurred and filter noise from real detections.

## Key Insight: Multi-Pass Confirmation

The CNN runs independently at each SAR acquisition date with temporal weighting (tau). At short tau (e.g. 6 days), each date's inference uses mostly independent SAR pairs. So if a pixel is above threshold at multiple consecutive dates, that means **multiple independent orbit passes** independently detected the debris.

- **Real debris**: Persists physically, so multiple passes see it. Bump width >= 3 at tau=6 means 3+ independent confirmations.
- **Noise / single-pass spikes**: Only one acquisition date fires (one weird SAR pair). Width=1 at tau=6. Longer tau can smear this single detection into a fake multi-step bump, which is misleading.

This is why **bump width at short tau** is the most important quality metric, not bump shape or symmetry.

## Scripts

### `temporal_onset.py` -- Main onset detector

Finds probability peaks, measures multi-pass confirmation, computes spatial context, outputs a NetCDF with per-pixel onset timing and quality metrics.

```bash
conda run -n sarvalanche python scripts/temporal_classifier/temporal_onset.py \
    --cnn-nc local/issw/tau_testing/tau6/season_v2_debris_probabilities.nc \
    --sar-nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
    --threshold 0.5 \
    --min-bump-width 2 \
    --spatial-radius 3
```

Can also be imported and called directly:
```python
from temporal_onset import run_temporal_onset
result = run_temporal_onset(cnn_ds, sar_ds, threshold=0.5)
```

**Output variables:**

| Variable | Description |
|---|---|
| `onset_date` | Estimated avalanche date (at peak probability) |
| `onset_step_idx` | Index into time axis of peak |
| `peak_prob` | Maximum CNN probability reached |
| `bump_width` | Contiguous steps above threshold (multi-pass confirmation) |
| `n_above_threshold` | Total steps above threshold (not necessarily contiguous) |
| `mean_detection_prob` | Mean probability across above-threshold steps |
| `persistence_ratio` | Fraction of first-to-last detection span with prob > 0.3 |
| `bump_smoothness` | Temporal smoothness (low 2nd-derivative energy) |
| `step_height_vv/vh` | VV/VH backscatter change at peak (dB) |
| `confidence` | 0-1 composite score (heavily weights multi-pass confirmation) |
| `spike_flag` | True if bump_width < min_bump_width (single-pass noise) |
| `spatial_bump_amplitude` | Gaussian-smoothed neighborhood prob at peak vs baseline |
| `spatial_peak_alignment` | Does neighborhood peak match pixel peak timing |
| `spatial_bump_symmetry` | Symmetry of spatial probability rise/fall |
| `pre_existing` | Debris present from first observation |
| `candidate_mask` | Pixels ever above threshold |

### `tau_comparison.py` -- Compare tau values

Runs temporal onset across tau=6,12,18,24,32 and outputs comparison CSV + plots.

```bash
conda run -n sarvalanche python scripts/temporal_classifier/tau_comparison.py
```

Outputs to `local/issw/tau_testing/`:
- `tau_comparison.csv` -- summary metrics per tau
- `tau_comparison.png` -- 6-panel comparison plot
- `tau_per_date_detections.png` -- detections per onset date by tau

### `plot_bump_examples.py` -- Visualize example pixels

Plots temporal probability time series and spatial evolution for 5 example pixels across tau=6,12,18.

```bash
conda run -n sarvalanche python scripts/temporal_classifier/plot_bump_examples.py
```

Outputs to `local/issw/tau_testing/`:
- `bump_examples.png` -- 5x3 grid of temporal time series
- `spatial_evolution_PtN_tauT.png` -- spatial maps at +/-4 steps around peak

## Prerequisites

Requires CNN probability cubes from Stage 1. To generate those:

```bash
# Run full-season CNN inference for a given tau
conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/full_season_inference.py \
    --nc local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc \
    --weights local/issw/v2_patches/v2_detector_best.pt \
    --season 2024-2025 \
    --tau 6 \
    --out-dir local/issw/tau_testing/tau6
```

## Design Decisions and Lessons Learned

1. **Bump shape metrics (skewness, rise sharpness, fall rate) were misleading.** We initially thought real debris would show a sharp rise then gradual fall. In practice, the bump shape is dominated by the tau smoothing effect, not the physical signal. A single-pass spike at tau=6 becomes a "nice smooth bump" at tau=18 purely due to temporal weight spreading.

2. **Multi-pass confirmation is the real quality signal.** If a pixel is detected at 3+ consecutive acquisition dates at short tau, those are genuinely independent SAR passes confirming the debris. This is far more discriminative than any bump shape metric.

3. **Longer tau artificially inflates width.** Comparing bump_width across tau values is not apples-to-apples. Width=4 at tau=6 (4 independent passes) is much stronger evidence than width=4 at tau=18 (could be 1 real pass smeared out). Always evaluate quality at the shortest tau available.

4. **The tau bug.** `full_season_inference.py` and `inference_scene.py` had a bug where the user-specified tau was not passed through to `calculate_empirical_backscatter_probability`, which had its own hardcoded `tau_days=6.0` default. Fixed by adding `tau_days=tau_days` to the call. Always verify tau propagates end-to-end.

5. **Spike rate at tau=6 is ~40%.** Nearly half of all candidate pixels are single-pass detections. The `spike_flag` (bump_width < 2) is the first-order noise filter. The ~30% of pixels with width >= 3 are the high-confidence detections.
