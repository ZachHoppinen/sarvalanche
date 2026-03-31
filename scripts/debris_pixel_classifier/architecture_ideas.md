# Architecture ideas for pairwise debris detector

Based on Gatti et al. (2026) "Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection" and our own training experiments.

## 1. SAR-only model (drop static channels)

Gatti et al.'s central finding: removing terrain auxiliaries (LIA, slope) gave equal or slightly better performance than the multimodal variant. Their best model uses only pre/post VV+VH — no DEM, no slope, no LIA.

**Experiment:** Train with N_INPUT=4 (change_vv, change_vh, change_cr, anf) dropping all 5 static channels (slope, aspect_northing, aspect_easting, cell_counts, tpi). If performance matches, it simplifies the pipeline significantly — no need for `build_static_stack`, `prepare_netcdf.py` derived vars, or TPI computation.

**Counterpoint:** Our static channels include cell_counts (FlowPy runout) which encodes avalanche path physics not available in SAR alone. This may be more informative than generic slope/LIA. Test with and without cell_counts specifically.

## 2. Siamese pre/post input (instead of pixel-level diffs)

Gatti et al. feed raw pre-event and post-event VV+VH as separate inputs to weight-sharing Swin Transformer encoders, then compute element-wise feature differences at the deepest encoder stage. This lets the model learn what features to diff.

**Our approach:** We compute dB diffs at the pixel level (sign_log1p(post - pre)) before the model sees them. This prescribes subtraction and discards absolute backscatter context.

**Experiment:** Feed 4 input channels (pre_VV, pre_VH, post_VV, post_VH) to a Siamese U-Net with shared encoder weights. Feature subtraction at bottleneck. The model learns what change representations matter, not just pixel diffs.

**Implementation:** New model class `SiamesePairDetector`. Requires changes to dataset (`__getitem__` returns pre+post patches instead of diffs) and inference pipeline.

## 3. Global SAR normalization

Gatti et al. clip SAR to [-40, 20] dB, normalize per-channel with dataset-wide mean and std, and use a sentinel value (normalized -50 dB) for invalid pixels. This provides a consistent input distribution across all scenes.

**Our approach:** We apply sign_log1p to dB diffs with no global normalization. The input distribution varies by scene, season, and pair span.

**Experiment:** Compute dataset-wide mean/std for each SAR channel across all training pairs. Apply z-score normalization. Use a sentinel value for invalid pixels instead of zero-fill.

## 4. More augmentation

Gatti et al. use:
- Geometric: horizontal flip, 90-degree rotations, mild affine (rotation, scale, shear)
- Radiometric: Gaussian noise + random gain perturbations on SAR channels
- Applied consistently to all inputs and mask

**Our approach:** Horizontal flip only.

**Priority augmentations to add:**
1. 90-degree rotations (need to rotate aspect_northing/easting channels to match)
2. Random gain perturbation on SAR channels (multiply by 1 +/- 0.1)
3. Gaussian noise on SAR channels (calibrated to denoised signal distribution)

## 5. F2-optimized threshold for recall

Gatti et al. show that F2-based threshold tuning (4x weight on recall) doubles the hit rate for size-2 avalanches (32% -> 64%) at the cost of more false positives.

**Application:** Our temporal onset pipeline could use an F2-tuned threshold for screening mode (flag all candidates) vs F1-tuned for mapping mode (conservative detections). Add `--threshold-metric f1|f2` to the inference/evaluation scripts.

## 6. Morphological post-processing

Gatti et al. apply dilation then erosion (3x3, 1 iteration) on the binary mask to connect fragmented detections and fill small gaps.

**Application:** Add as an optional post-processing step after thresholding in the temporal onset pipeline. Simple and effective for cleaning up noisy pixel-level predictions into coherent debris regions.

## 7. Swin Transformer backbone

Gatti et al. use Swin Transformer V2 Tiny (2.39M params after simplification) instead of a CNN. They found it outperformed SiamUnet, STANet, BIT, SNUNet-CD, ChangeFormer, TinyCD, and STNet on their avalanche test set (F1=0.803, IoU=0.661).

**Our model:** SinglePairDetector U-Net with 928K params. Smaller but CNN-based.

**Experiment:** Replace U-Net backbone with Swin Transformer. Higher compute cost but potentially better spatial reasoning for long-range debris path patterns. Consider TinyCD (0.29M params, F1=0.766) as a lightweight alternative that still outperformed most baselines.

## Priority order

1. SAR-only experiment (quick, tests a fundamental assumption)
2. More augmentation (medium effort, helps generalization)
3. Siamese pre/post input (bigger refactor but potentially significant)
4. Global normalization (dataset preprocessing change)
5. F2 threshold + morphological post-processing (inference-only changes)
6. Swin Transformer backbone (architecture replacement, most effort)
