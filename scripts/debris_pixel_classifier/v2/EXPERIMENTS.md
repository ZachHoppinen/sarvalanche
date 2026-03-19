# v2 CNN Debris Detector — Experiment Log

## Architecture

The v2 CNN uses a **set encoder + spatial attention** architecture:
- **SAR input**: Variable number of crossing pairs per date, each (C, 128, 128)
- **Set encoder**: Shared-weight CNN encodes each pair to feature maps
- **Attention pooling**: Learned attention weights combine pairs into a single representation
- **Static encoder**: Separate CNN for terrain/auxiliary channels
- **Fusion + Decoder**: Concatenate SAR + static features → 4-stage ConvTranspose decoder → (1, 128, 128) probability

Two variants tested: 3-channel SAR (change, ANF, proximity) and 4-channel SAR (+ melt_weight).

## Data Sources

### SAR
- Sentinel-1 OPERA RTC-S1 products at 30m resolution
- Season NetCDFs for Turnagain Pass and Girdwood, 2024-2025 and 2025-2026
- VV and VH polarizations, multiple orbit tracks (65, 131, 160)

### Human Labels
- **Feb 14, 2026**: 673 polygons (new, expanded labels)
- **Dec 15, 2025**: 216 polygons (new)
- **Feb 14, 2026 (old)**: 143 polygons (original labels)

### Auto Labels
- Smart auto-labeler with elevation-banded Otsu thresholding
- Cross-ratio wet-snow suppression
- FlowPy runout boosting
- Scene-wide melt detection (B6)

### Observation Data (Validation)
- **AKDOT**: 185 observations in scene, avalanche paths (114 polygons)
- **AKRR**: 34 observations, paths (43 polygons)
- **Temporal split**: 60/40 by date → 132 train / 87 val observations
- **Split date**: 2026-01-28

### Auxiliary
- HRRR-Alaska 3km surface temperature (8 hours/day, lapse-rate adjusted to 30m DEM)
- RGI 7.0 glacier outlines (331 glaciers, 12% of scene)
- FlowPy runout modeling (cell_counts, release_zones, runout_angle)

## Experiments and Results

All validation against held-out observations after 2026-01-28 (87 obs, 85 matched to paths).

### Experiment 1: SNFAC Pretrain (3-stage pipeline)

**Hypothesis**: Pretraining on Idaho (SNFAC) human labels transfers useful debris features to AK.

| Model | Det@0.2 | Det@0.5 | F1@0.2 | FPR@0.2 |
|-------|---------|---------|--------|---------|
| human_only (AK, 143 labels) | 35/85 (41%) | 12/85 (14%) | 0.166 | 22.8% |
| 3stage (SNFAC→bridge→ft) | 28/85 (33%) | 25/85 (29%) | 0.129 | 24.2% |
| 2stage (AK auto→human) | 1/85 (1%) | 1/85 (1%) | 0.029 | 12.0% |
| human_snfac_ft | 3/85 (4%) | 3/85 (4%) | 0.067 | 17.1% |
| human_combined (SNFAC+AK) | 7/85 (8%) | 5/85 (6%) | 0.054 | 32.6% |

**Finding**: SNFAC pretrain consistently hurts. The domain gap between Idaho (continental snow) and AK (maritime snow) is too large. The 3stage model over-detects — 5x more pixels on quiet days than human_only. human_only with just 143 AK labels outperforms everything.

### Experiment 2: HRRR Melt Filtering

**Hypothesis**: Downweighting SAR pairs acquired during warm conditions removes melt false positives.

Key findings from d_empirical analysis:
- **Feb 13**: 75% of >3dB pixels were melt noise from Feb 1/3 warm spell (650k → 165k pixels)
- **Dec 15**: 13% removed (less melt contamination, pairs reach back to Dec 3 warm spell)
- **Feb 25**: Residual melt artifacts at low elevation where HRRR Tmax ≈ -1 to -4°C (solar melt not captured)

Melt weight implementation:
- PDD-based: `weight = clip(1 - PDD/0.1, 0, 1)` — any positive degree-day suppresses
- Tmax-based: `weight = clip((-Tmax - 3) / 5, 0, 1)` — suppress above -3°C
- Combined: `min(PDD_weight, Tmax_weight)` per pair endpoint
- Gaussian smoothed (sigma=15px ≈ 450m) to avoid 3km HRRR grid artifacts

### Experiment 3: Melt Features as CNN Channels

**Hypothesis**: Giving the CNN melt information (4th SAR channel + filtered/residual static channels) improves discrimination.

| Model | Labels | SAR ch | Static ch | Det@0.2 | Det@0.5 | F1@0.2 | FPR@0.2 |
|-------|--------|--------|-----------|---------|---------|--------|---------|
| human_only (orig) | 143 | 3 | 11 | 41% | 14% | 0.166 | 22.8% |
| human_3ch (13st) | 150 | 3 | 13 | 15% | 8% | 0.116 | 12.0% |
| human_4ch (150 lbl) | 150 | 4 | 13 | 11% | 4% | 0.062 | 19.0% |
| **human_4ch (889 lbl)** | **889** | **4** | **13** | **44%** | **24%** | **0.200** | **14.5%** |

**Finding**: With only 150 labels, the extra channels hurt (model can't learn them). With 889 labels (6x more), the 4ch/13st model is the best overall: F1=0.200 (+20% over original), FPR=14.5% (36% fewer false positives), D3 detection=100%.

### Experiment 4: Auto-labeling Improvements

Tested multiple auto-labeling enhancements:
- Manual d_empirical thresholds on melt-filtered signal (3.0-6.5 dB by date)
- Size filter: 5,000-750,000 m² (matched human label distribution)
- Mean cell_count ≥ 50 (FlowPy runout requirement)
- Glacier mask: reject >20% overlap with RGI 7.0
- Water mask: reject >10% overlap

| Filter stage | Feb 13 components | Dec 15 components |
|---|---|---|
| Total | 21,577 | 12,725 |
| After size filter | 2,801 | 1,174 |
| After glacier filter | 2,547 (-254) | 962 (-212) |
| After runout filter | 1,605 (-942) | 566 (-396) |

**Finding**: Auto-labeling improvements produced cleaner labels, but auto_ft models still massively over-detect (94% detection but 54% FPR). Auto-labeling dropped in favor of human-only approach.

### Experiment 5: Morphology-Based Confidence Scoring

Added 5 spatial checks to `assign_confidence.py`:
1. **Coverage ratio**: debris fraction of valid terrain (>40% → 0.1 confidence)
2. **Elevation concentration**: debris at top of patch → suspicious
3. **FlowPy runout overlap**: >30% overlap → terrain-plausible
4. **Slope uniformity**: uniform steep slope → likely melt
5. **Spatial coherence**: >40% of patch bright → scene-wide melt

Also: smooth continuous cross-ratio penalty replacing 2-tier step function.

**Finding**: Not independently validated, but integrated into all training pipelines.

## Key Findings

1. **More human labels >> any architecture change**. Going from 143 → 889 labels gave +20% F1. This is the single biggest lever.

2. **SNFAC pretrain hurts**. The domain gap between Idaho and AK is too large for transfer learning to help at any stage.

3. **Auto-labeling consistently hurts**. Every auto-label pretrain variant performs worse than human-only, despite extensive melt filtering, size/runout/glacier filtering, and manual thresholds.

4. **HRRR melt filtering is very effective at the d_empirical level**. Removing warm-contaminated pairs reduces false positives by 50-75% on melt-affected dates.

5. **The melt_weight CNN channel helps with enough labels**. At 150 labels it hurts (can't learn the channel), at 889 it contributes to the best model.

6. **D-size scaling is physically sensible**. The best model detects D3 avalanches at 100% and scales down appropriately (D2.5: 67%, D2: 46%, D1.5: 57%, D1: 29%).

7. **The v2 attention architecture is complex and hard to debug**. Single-pair evaluation is proposed for v3.

## Best Model

**human_only_4ch** (AK v2 experiment):
- 889 human labels (Dec 15 + Feb 14)
- 4 SAR channels (change, ANF, proximity, melt_weight)
- 13 static channels (terrain + melt-filtered d_empirical + residual)
- F1@0.2 = 0.200, FPR@0.2 = 14.5%
- D3: 100%, D2.5: 67%, D2: 46% detection at threshold 0.2
- Weights: `local/cnfaic/ak_v2_experiment/weights/human_only_4ch.pt`

## Next Steps → v3

Moving to a **single-pair U-Net** architecture:
- Each SAR crossing pair evaluated independently (no attention/set encoder)
- 7 SAR channels per pair: change, ANF, proximity, melt_weight, VV magnitude, VH magnitude, per-pair cross-ratio
- 13 static channels: terrain + curvature + TPI + melt-filtered d_empirical + d_cr
- Each labeled patch × ~24 pairs = ~21k training samples (vs 889 in v2)
- Temporal aggregation via existing `temporal_onset.py` post-inference
- Validation paths held out from training, used as ground truth
- See `src/sarvalanche/ml/v3/` for implementation
