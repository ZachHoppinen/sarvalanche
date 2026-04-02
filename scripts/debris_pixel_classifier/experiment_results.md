# Pairwise Debris Classifier — Experiment Results

All runs use 27 zones, 133k samples, same val split (seed 42), 150 epochs,
auto-label-frac 0.25, curriculum (top 20%/50%/all at epochs 0/10/20).

## Run 1: Full model (9ch, h-flip only) — BASELINE

- **Config**: 4 SAR (change_vv, change_vh, change_cr, anf) + 5 static (slope, aspect_northing, aspect_easting, cell_counts, tpi)
- **Pretrained from**: combined_best.pt (epoch 57, val_loss=0.4885)
- **Output**: `all_sites_best.pt`
- **Best epoch**: 115 (tied with 112)
  - val_loss: **0.3663**
  - F1@0.5: **0.691**
  - IoU@0.5: **0.528**
  - P@0.5: 0.612, R@0.5: 0.792
- **Notes**: Strong recall. Pretrained weights helped fast convergence.

## Run 2: Full model + v-flip augmentation

- **Config**: Same 9ch, added vertical flip with aspect_northing negation
- **Pretrained from**: all_sites_best.pt (Run 1)
- **Output**: `all_sites_v2_best.pt`
- **Best epoch**: 135
  - val_loss: **0.4496**
  - F1@0.5: **0.623**
  - IoU@0.5: **0.452**
  - P@0.5: 0.551, R@0.5: 0.715
- **Conclusion**: V-flip hurts significantly (~10% F1 drop). SAR range-direction orientation is real signal, not overfitting. **Dropped v-flip, kept h-flip only.**

## Run 3: SAR-only (4ch, h-flip only) — STOPPED

- **Config**: 4 SAR channels only (change_vv, change_vh, change_cr, anf), no static channels
- **Pretrained from**: None (fresh start)
- **Output**: `sar_only_best.pt`
- **Stopped at epoch 42**:
  - val_loss: 0.5587
  - F1@0.5: 0.531
  - IoU@0.5: 0.362
  - P@0.5: 0.420, R@0.5: 0.722
- **Conclusion**: ~25% behind full model. 2x more predicted pixels (low precision without terrain context). Static channels — especially FlowPy cell_counts — provide spatial precision that SAR alone cannot. Gatti et al.'s finding that static channels don't help was likely because they used generic LIA/slope, not domain-specific avalanche path modeling.

## Summary Table (best results)

| Run | Channels | F1@0.5 | IoU@0.5 | P@0.5 | R@0.5 | val_loss |
|-----|----------|--------|---------|-------|-------|----------|
| 1. Full (baseline) | 9 (4 SAR + 5 static) | **0.691** | **0.528** | 0.612 | 0.792 | 0.3663 |
| 2. Full + v-flip | 9 + v-flip aug | 0.623 | 0.452 | 0.551 | 0.715 | 0.4496 |
| 3. SAR-only | 4 (SAR only) | 0.531 | 0.362 | 0.420 | 0.722 | 0.5587 |

Run 3 stopped at epoch 42/150 — clear that static channels (esp. FlowPy cell_counts) are essential.

## Run 4: Siamese U-Net (shared encoder, feature-level diff)

- **Config**: 3 branch channels per branch (VV dB, VH dB, ANF normalized) + 5 static injected at decoder
- **Architecture**: Weight-sharing encoder on pre/post patches, element-wise feature subtraction at each level, static channels at full-res decoder stage. 928K params.
- **Pretrained from**: None (fresh start)
- **Output**: `siamese_debris_detector/best.pt`
- **Best epoch**: 145
  - val_loss: **0.4452**
  - F1@0.5: **0.628**
  - IoU@0.5: **0.457**
  - P@0.5: 0.526, R@0.5: 0.779
- **Issues**: Occasional training collapses (epochs 38, 76, 131, 138, 147) where recall crashes to ~0.2 briefly. Architecture less stable than pairwise.
- **Conclusion**: ~10% behind pairwise on F1. The hand-crafted `sign_log1p(dB_diff)` representation outperforms learned feature differencing at this model capacity (928K params). Siamese might benefit from more capacity or attention-based fusion, but pairwise diff approach is better for our U-Net size.

## Summary Table (best results)

| Run | Architecture | Channels | F1@0.5 | IoU@0.5 | P@0.5 | R@0.5 | val_loss |
|-----|-------------|----------|--------|---------|-------|-------|----------|
| 1. Pairwise (baseline) | U-Net | 9 (4 SAR diff + 5 static) | **0.691** | **0.528** | 0.612 | 0.792 | 0.3663 |
| 2. Pairwise + v-flip | U-Net | 9 + v-flip aug | 0.623 | 0.452 | 0.551 | 0.715 | 0.4496 |
| 3. SAR-only | U-Net | 4 (SAR diff only) | 0.531 | 0.362 | 0.420 | 0.722 | 0.5587 |
| 4. Siamese | Siamese U-Net | 3+3 branch + 5 static | 0.628 | 0.457 | 0.526 | 0.779 | 0.4452 |

Run 2 used pretrained weights (unfair advantage on convergence speed, but lower ceiling due to v-flip).
Run 3 stopped at epoch 42.

## Run 5: Pairwise fresh baseline (9ch, h-flip, no pretrain)

- **Config**: Same as Run 1 but no pretrained weights — trains from scratch
- **Pretrained from**: None
- **Output**: `baseline_fresh_best.pt`
- **Best epoch**: 127
  - val_loss: **0.4168**
  - F1@0.5: **0.650**
  - IoU@0.5: **0.482**
  - P@0.5: 0.575, R@0.5: 0.749
- **Conclusion**: ~6% behind Run 1 on F1 (0.650 vs 0.691). The pretrained weights in Run 1 gave a better starting point that the fresh run couldn't fully close in 150 epochs. Recall is lower (0.749 vs 0.792), precision similar.

## Run 6: TinyCD (Siamese, learned mixing, 343K params)

- **Config**: 3 branch channels per branch (VV dB, VH dB, ANF) + 5 static. Depthwise separable convs, squeeze-excite attention, Mix modules for learned feature fusion. base_ch=24.
- **Pretrained from**: None (fresh start)
- **Output**: `tinycd_debris_detector/best.pt`
- **Best epoch**: 143
  - val_loss: **0.5882**
  - F1@0.5: **0.509**
  - IoU@0.5: **0.341**
  - P@0.5: 0.384, R@0.5: 0.752
- **Issues**: Same occasional collapse as Siamese (epoch 147). Plateaued around F1@0.5=0.50 from epoch ~120 onward.
- **Conclusion**: Worst performer. 343K params is too small for this task — the efficient architecture couldn't compensate for the capacity deficit. The Mix module's learned fusion didn't outperform the Siamese subtraction (Run 4 got F1=0.628 with 928K params). Confirms that raw pre/post inputs are fundamentally harder than explicit diffs, and smaller models can't bridge that gap.

## Run 7: V2 — larger model + post context + dropout (11ch, 3.7M params)

- **Config**: 11 channels (4 diff + 2 post VV/VH + 5 static), base_ch=32 (3.7M params), dropout=0.1
- **Pretrained from**: None (fresh start)
- **Output**: `v2_best.pt`
- **Best epoch**: 310
  - val_loss: **0.2798**
  - F1@0.5: **0.766**
  - IoU@0.5: **0.621**
  - P@0.5: 0.672, R@0.5: 0.891
- **Train/val gap**: train=0.15, val=0.28 (1.9x — healthy)
- **Notes**: Plateaued around epoch 280. Extended to 350 epochs. The post VV/VH channels + larger model + dropout all contributing. Best result across all experiments.

## Updated Summary Table

| Run | Architecture | Params | Channels | Pretrain | F1@0.5 | IoU@0.5 | P@0.5 | R@0.5 | val_loss |
|-----|-------------|--------|----------|----------|--------|---------|-------|-------|----------|
| 1. Pairwise (pretrained) | U-Net | 928K | 9 (4 SAR diff + 5 static) | Yes | **0.691** | **0.528** | 0.612 | **0.792** | **0.3663** |
| 5. Pairwise (fresh) | U-Net | 928K | 9 (4 SAR diff + 5 static) | No | 0.650 | 0.482 | 0.575 | 0.749 | 0.4168 |
| 4. Siamese | Siamese U-Net | 928K | 3+3 branch + 5 static | No | 0.628 | 0.457 | 0.526 | 0.779 | 0.4452 |
| 2. Pairwise + v-flip | U-Net | 928K | 9 + v-flip aug | Yes | 0.623 | 0.452 | 0.551 | 0.715 | 0.4496 |
| 3. SAR-only | U-Net | 928K | 4 (SAR diff only) | No | 0.531 | 0.362 | 0.420 | 0.722 | 0.5587 |
| 6. TinyCD | TinyCD | 343K | 3+3 branch + 5 static | No | 0.509 | 0.341 | 0.384 | 0.752 | 0.5882 |
| **7. V2 (kitchen sink)** | **U-Net** | **3.7M** | **11 (4 diff + 2 post + 5 static)** | **No** | **0.766** | **0.621** | **0.672** | **0.891** | **0.2798** |

Runs sorted by F1@0.5. Run 3 stopped at epoch 42.

## Run 8: V3 — base_ch=48, 250 epochs

- **Config**: 11 channels (4 diff + 2 post VV/VH + 5 static), base_ch=48 (8.3M params), h-flip only
- **Pretrained from**: None (fresh start)
- **Output**: `v3_best.pt` → `pairwise_debris_detector_v0.1.dev48+ge87ef02fa.d20260210_20260402.pth`
- **Best epoch**: 242 (of 250)
  - val_loss: **0.2673**
  - F1@0.5: **0.778**
  - IoU@0.5: **0.637**
  - P@0.5: 0.678, R@0.5: 0.914
- **Train/val gap**: train=0.14, val=0.27 (1.9x — healthy)
- **Notes**: Steady improvement through full 250 epochs, LR decayed from 9.1e-4 to 3.2e-4. Higher capacity (8.3M vs 3.7M) and 250 epochs pushed past v2's ceiling. Recall improved substantially over v2 (0.914 vs 0.891) while maintaining similar precision.
- **Conclusion**: New best. +1.2% F1, +1.6% IoU over v2 with 2.2× more parameters. Still converging slowly at epoch 250 — longer training or larger LR warmup could push further.

## Updated Summary Table

| Run | Architecture | Params | Channels | Pretrain | F1@0.5 | IoU@0.5 | P@0.5 | R@0.5 | val_loss |
|-----|-------------|--------|----------|----------|--------|---------|-------|-------|----------|
| **8. V3 (base_ch=48)** | **U-Net** | **8.3M** | **11 (4 diff + 2 post + 5 static)** | **No** | **0.778** | **0.637** | **0.678** | **0.914** | **0.2673** |
| 7. V2 (base_ch=32) | U-Net | 3.7M | 11 (4 diff + 2 post + 5 static) | No | 0.766 | 0.621 | 0.672 | 0.891 | 0.2798 |
| 1. Pairwise (pretrained) | U-Net | 928K | 9 (4 SAR diff + 5 static) | Yes | 0.691 | 0.528 | 0.612 | 0.792 | 0.3663 |
| 5. Pairwise (fresh) | U-Net | 928K | 9 (4 SAR diff + 5 static) | No | 0.650 | 0.482 | 0.575 | 0.749 | 0.4168 |
| 4. Siamese | Siamese U-Net | 928K | 3+3 branch + 5 static | No | 0.628 | 0.457 | 0.526 | 0.779 | 0.4452 |
| 2. Pairwise + v-flip | U-Net | 928K | 9 + v-flip aug | Yes | 0.623 | 0.452 | 0.551 | 0.715 | 0.4496 |
| 3. SAR-only | U-Net | 928K | 4 (SAR diff only) | No | 0.531 | 0.362 | 0.420 | 0.722 | 0.5587 |
| 6. TinyCD | TinyCD | 343K | 3+3 branch + 5 static | No | 0.509 | 0.341 | 0.384 | 0.752 | 0.5882 |

Runs sorted by F1@0.5. Run 3 stopped at epoch 42.
