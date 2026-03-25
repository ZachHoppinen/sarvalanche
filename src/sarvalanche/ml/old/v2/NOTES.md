# V2 Debris Detection CNN — Full Pipeline Notes

## Table of Contents

1. [Overview](#overview)
2. [Getting Patches to Label](#getting-patches-to-label)
3. [Labeling Workflow](#labeling-workflow)
4. [Extracting Training Patches from Labels](#extracting-training-patches-from-labels)
5. [Training](#training)
6. [Inference](#inference)
7. [Model Architecture Deep Dive](#model-architecture-deep-dive)
8. [Data Format Reference](#data-format-reference)
9. [Key Constants & Hyperparameters](#key-constants--hyperparameters)

---

## Overview

The v2 CNN is a **single-pass debris detector** that takes a variable number of per-track/polarization SAR backscatter change maps plus 8 static terrain channels and outputs a pixel-level debris probability map at 128x128 resolution.

Previous design (deprecated): Two-pass architecture where Pass1 learned normalized patterns and Pass2 fused magnitude + terrain + Pass1 confidence. Pass1 never learned (val_loss stuck, IoU=0.0 for 30 epochs) while Pass2 did all the work (IoU=0.38). Simplified to single pass — essentially Pass2 without the ConfidenceEncoder.

### Pipeline at a glance

```
season_dataset.nc
    → [patch_labeler.py --save-geotiff]  export GeoTIFFs to QGIS
    → [QGIS] draw debris polygons → avalanche_labels.gpkg
    → [extract_patches_from_polygons.py]  rasterize polygons → 128x128 .npz patches
    → [train.py]  train DebrisDetector
    → [inference_scene.py]  sliding-window whole-scene prediction → GeoTIFF
```

---

## Getting Patches to Label

### Step 1: Generate GeoTIFF windows for visual review

The `patch_labeler.py` script selects ~10 km windows from the scene — half from high-probability areas, half from low-probability areas — and exports them as georeferenced GeoTIFFs that can be opened in QGIS for polygon labeling.

```bash
conda run -n sarvalanche python scripts/manual_labeling/patch_labeler.py \
    --nc 'local/issw/dual_tau_output/zone/season_dataset.nc' \
    --date 2025-02-04 \
    --tau 6 \
    --n-patches 100 \
    --patch-km 10 \
    --save-geotiff
```

This produces:
- `patches/<date>/geotiffs/<date>/d_empirical_<date>_tau6_<bucket>_<y>_<x>.tif` — d_empirical for each window
- `patches/<date>/patch_footprints.gpkg` — footprint polygons for every exported window (has columns: patch_id, date, mean_p, bucket, geotiff, reviewed)
- `patches/<date>/avalanche_labels_<date>.gpkg` — seed GeoPackage with one tiny polygon so QGIS recognizes it as a polygon layer

#### How window selection works

1. Scene is tiled into non-overlapping windows of size `window_px` (computed from `--patch-km`)
2. Mean `p_empirical` is computed per window; windows with <50% valid pixels are dropped
3. Half the highest-scoring and half the lowest-scoring windows are selected
4. They're interleaved (high, low, high, low...) for balanced presentation

### Alternative: Interactive matplotlib labeling

If you skip `--save-geotiff`, the script opens a matplotlib UI for binary labeling:
- Keys: `1` = debris, `0` = no debris, `n` = skip, `←`/`→` = navigate, `q` = quit
- Shows 4 panels: d_empirical, p_empirical, slope, cell_counts
- When `--v2` flag is set, each labeled ~10km window is immediately tiled into 128x128 v2 patches at stride=32

### Alternative: Full season track-level labeler

For track-level labeling across the full season (all dates, all tracks):

```bash
conda run -n sarvalanche python scripts/manual_labeling/full_season_manual_labeler.py \
    --season-nc local/issw/dual_tau_output/zone/season_dataset.nc \
    --tracks-gpkg local/issw/dual_tau_output/zone/season_tracks.gpkg \
    --zone Sawtooth_&_Western_Smoky_Mtns
```

Features:
- Rotates to a random date every 50 labels
- Optional XGBoost-based active learning: scores unlabeled tracks and prioritizes uncertain ones (60% uncertain, 20% random, 10% tail, 10% unscored)
- Biases towards larger tracks (~80% large, 20% small)
- Interactive polygon/lasso drawing directly on the matplotlib figure (`d` to enter draw mode, `p` = polygon, `l` = lasso, `z` = undo)
- 4-level labels: 0 = no, 1 = unsure no, 2 = unsure yes, 3 = yes
- Drawn shapes saved to `debris_shapes.gpkg`
- Nearby SNFAC avalanche observations shown as cyan dots within ±6 days

---

## Labeling Workflow

### Recommended workflow: GeoTIFF → QGIS → Polygon extraction

This is the most precise labeling method — you draw polygons directly on georeferenced imagery.

1. **Export GeoTIFFs** (see above)
2. **Open in QGIS**:
   - Load the `d_empirical_*.tif` files — these show backscatter change (blue = decrease, red = increase)
   - Load the `avalanche_labels_<date>.gpkg` — this is your polygon layer
   - Load `patch_footprints.gpkg` to see which windows were exported
3. **Draw polygons** around debris features in the avalanche_labels layer:
   - Look for spatially coherent patches of high d_empirical (blue/dark) that follow realistic runout paths
   - Cross-reference with slope, cell_counts if loaded
   - Each polygon = one avalanche debris field
4. **Mark footprints as reviewed**:
   - In `patch_footprints.gpkg`, set `reviewed = True` for windows you've inspected
   - This ensures negative patches (no polygon overlap) are only extracted from areas you actually looked at, preventing false negatives from unreviewed areas

### What to look for when labeling

- **d_empirical < -0.5**: Strong backscatter decrease — potential wet snow/debris deposition
- **Spatial coherence**: Real avalanche debris follows terrain — starts at release zone (steep slope), flows through track, deposits at runout
- **Size**: Typical debris deposits are 100-1000+ meters long, narrow in cross-track
- **Slope context**: Release zones typically 30-50°, deposition zones 15-30°
- **cell_counts > 0**: Pixels in FlowPy-modeled runout zones are more likely to have debris

### What NOT to label

- Isolated single pixels of high d_empirical (noise)
- Changes over water bodies or glaciers
- Forest cover changes (different temporal signature)
- Road/infrastructure changes

---

## Extracting Training Patches from Labels

Once you have debris polygons drawn in QGIS, extract 128x128 training patches:

```bash
conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
    --nc 'local/issw/dual_tau_output/zone/season_dataset.nc' \
    --polygons local/issw/debris_shapes/avalanche_labels_2025-02-04.gpkg \
    --footprints local/issw/debris_shapes/patch_footprints.gpkg \
    --date 2025-02-04 \
    --tau 6 \
    --out-dir local/issw/v2_patches/2025-02-04 \
    --stride 64 \
    --neg-ratio 3.0 \
    --min-debris-frac 0.005
```

### What it does

1. **Loads season_dataset.nc** and computes empirical layers for the target date:
   - Temporal weights with tau=6 days
   - Per-track/pol `d_{track}_{pol}_empirical` (backscatter change maps)
   - Per-track/pol `p_{track}_{pol}_empirical` (empirical probabilities)
   - Combined `d_empirical` and `p_empirical` (agreement-boosted)

2. **Rasterizes debris polygons** to a binary (H, W) mask on the dataset grid:
   - Uses `rasterio.features.geometry_mask` with `all_touched=True`
   - Handles CRS reprojection automatically

3. **Determines extraction bounds** from footprints:
   - Only extracts patches within reviewed footprint windows
   - This is critical — areas outside footprints are NOT labeled, so extracting negatives from there would create false negatives
   - If `--footprints` not provided, uses full scene (only do this if you've reviewed the entire scene)

4. **Slides 128x128 windows** at `--stride` (default 64) within each footprint:
   - **Positive patch**: debris fraction ≥ `--min-debris-frac` (default 0.5% = ~82 pixels)
   - **Negative patch**: debris fraction < threshold
   - Negatives subsampled to `--neg-ratio` (default 3:1 neg:pos)

5. **For each patch**, calls `build_v2_patch(ds, y0, x0, 128)`:
   - Extracts per-track/pol SAR change maps from `d_{track}_{pol}_empirical` variables
   - Builds 8-channel static terrain stack (see [Data Format Reference](#data-format-reference))
   - Applies global normalization to all static channels except DEM
   - Applies per-patch min-max normalization to DEM
   - Skips patches with no SAR signal (max < 1e-6)

6. **Saves each patch** as compressed .npz:
   ```
   {pos|neg}_{y0:04d}_{x0:04d}_v2_.npz
   ```
   containing:
   - `sar_maps`: (N, 128, 128) float32 — N varies per scene (typically 2-8)
   - `static`: (8, 128, 128) float32 — normalized static terrain
   - `label`: int8 (0 or 1)
   - `label_mask`: (128, 128) float32 — pixel-level debris mask from polygons
   - `x_coords`, `y_coords`: coordinate arrays
   - `crs`: CRS string

7. **Saves metadata** to `labels.json`:
   ```json
   {
     "pos_0128_0256": {"label": 1, "y0": 128, "x0": 256, "debris_frac": 0.032},
     "neg_0064_0192": {"label": 0, "y0": 64, "x0": 192, "debris_frac": 0.0}
   }
   ```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stride` | 64 | Extraction stride in pixels. 64 = 50% overlap between 128px patches |
| `--neg-ratio` | 3.0 | Max negative:positive ratio. 3.0 means at most 3 neg per pos |
| `--min-debris-frac` | 0.005 | Min fraction of debris pixels for positive class. 0.5% of 128x128 = ~82 pixels |
| `--footprints` | None | GeoPackage limiting extraction to reviewed areas. Strongly recommended |

### Multi-date extraction

Run `extract_patches_from_polygons.py` separately for each date, outputting to different subdirectories:

```bash
for DATE in 2025-01-15 2025-02-04 2025-02-20; do
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
        --nc '..../season_dataset.nc' \
        --polygons "local/issw/debris_shapes/avalanche_labels_${DATE}.gpkg" \
        --footprints local/issw/debris_shapes/patch_footprints.gpkg \
        --date "$DATE" --tau 6 \
        --out-dir "local/issw/v2_patches/${DATE}"
done
```

The dataset loader (`V2PatchDataset`) recursively finds all `labels.json` files under `--data-dir`, so point it at the parent directory to train on all dates:

```bash
--data-dir local/issw/v2_patches
```

---

## Training

```bash
conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/train.py \
    --data-dir local/issw/v2_patches \
    --epochs 50 \
    --pos-weight 10 \
    --batch-size 4 \
    --lr 1e-3
```

### What happens

1. **Dataset loading**:
   - `V2PatchDataset` recursively searches `--data-dir` for `labels.json` files
   - For each window_id in labels.json, finds matching `{window_id}_v2_*.npz` files
   - Fallback: loads all `*_v2_*.npz` files directly if no labels.json found
   - Splits into train (85%) / val (15%) via `random_split`
   - Augmentation enabled on train set, disabled on val set

2. **Data augmentation** (train set only):
   - **Horizontal flip**: 50% chance, applied to SAR maps, static channels, and label mask
   - **Additive Gaussian noise**: σ=0.05 on each SAR map independently

3. **Collation** (`v2_collate_fn`):
   - Patches can have different numbers of SAR maps (different track/pol counts)
   - Shorter lists padded with zero tensors to match the max in the batch
   - Output: `sar_maps` = list of N tensors each (B, 1, 128, 128), `static` = (B, 8, 128, 128), `label` = (B, 1, 128, 128)

4. **Positive class weight**:
   - If `--pos-weight > 0`: use that value directly
   - If `--pos-weight 0` (default): auto-compute from data
     - Counts positive vs negative pixels across all patches
     - Formula: `pw = n_neg_pixels / n_pos_pixels`
     - Capped at 50.0 to prevent extreme values
   - Passed to `binary_cross_entropy_with_logits` as `pos_weight` tensor

5. **Training loop** (single phase):
   - **Optimizer**: Adam, lr=1e-3
   - **Scheduler**: CosineAnnealingLR(T_max=epochs) — lr decays to ~0 by end
   - **Loss**: `F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)`
   - Each epoch: forward → loss → backward → optimizer step
   - Gradient norms tracked per submodule: set_enc, attn, static_enc, fusion, dec

6. **Validation** (every epoch):
   - Unweighted BCE loss (no pos_weight)
   - IoU at threshold=0.5: `intersection / union`
   - Best model saved by lowest val_loss

7. **Checkpointing**:
   - Saves `state_dict` (weights only) to `v2_detector_best.pt`
   - Default location: `{data-dir}/v2_detector_best.pt`
   - Override with `--out`

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Total training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 8 | Batch size |
| `--val-frac` | 0.15 | Fraction held out for validation |
| `--pos-weight` | 0 (auto) | Positive class weight for BCE. 0 = auto-compute from data |
| `--out` | `{data-dir}/v2_detector_best.pt` | Output weights path |
| `--device` | auto | Force device (mps/cuda/cpu). Auto-detects if omitted |

### Logging

Every 10 epochs (and epoch 1):
```
epoch  10: train=0.3421  val=0.4102  IoU=0.1523  grads: set_enc=0.0234  attn=0.0012  static_enc=0.0189  fusion=0.0045  dec=0.0312
```

---

## Inference

Run the trained model over a full scene with sliding windows:

```bash
conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/inference_scene.py \
    --nc local/issw/dual_tau_output/zone/season_dataset.nc \
    --date 2025-02-04 \
    --tau 6 \
    --weights local/issw/v2_patches/v2_detector_best.pt \
    --stride 32 \
    --batch-size 16 \
    --out scene_v2_debris_2025-02-04.tif
```

### What happens

1. Loads `season_dataset.nc` into memory
2. Computes `w_resolution` if missing, then empirical layers for the target date
3. Builds per-track/pol SAR change maps + static terrain stack (scene-wide)
4. Slides 128x128 window at `--stride` (default 32 = 75% overlap):
   - Edge handling: ensures complete coverage by adding edge-aligned patches
   - Per-patch DEM normalization applied to static channels
5. Batches patches, runs model, applies sigmoid
6. Averages overlapping predictions: `prob_sum / count`
7. Saves (H, W) probability map as georeferenced GeoTIFF

### Output

- `scene_v2_debris_<date>.tif` — float32 probability map [0, 1]
- CRS and spatial coords inherited from input dataset

---

## Model Architecture Deep Dive

### DebrisDetector — top-level model

```
Inputs:
  sar_maps: list of N tensors, each (B, 1, 128, 128)  ← variable N
  static:   (B, 8, 128, 128)

Pipeline:
  sar_maps ──→ [SetEncoder shared] ──→ N × (B, 64, 8, 8)
                                            │
                                    [SpatialSetAttention]
                                            │
                                      (B, 64, 8, 8)  ← aggregated SAR features
                                            │
  static ────→ [StaticEncoder] ──→ (B, 32, 8, 8)
                                            │
                          ┌─────────────────┤
                          │ cat (dim=1)     │
                          ▼                 ▼
                     (B, 96, 8, 8)
                          │
                    [Conv2d 1x1: 96→64]
                    [ReLU]
                          │
                     (B, 64, 8, 8)
                          │
                      [Decoder]
                          │
                     (B, 1, 128, 128)  ← logits (apply sigmoid for probs)
```

### ConvBlock(in_ch, out_ch, stride=1)

Two-layer conv block. Used as the building block throughout.

```
Input: (B, in_ch, H, W)
  → Conv2d(in_ch, out_ch, 3x3, stride=stride, padding=1, bias=False)
  → BatchNorm2d(out_ch)
  → ReLU(inplace=True)
  → Conv2d(out_ch, out_ch, 3x3, stride=1, padding=1, bias=False)
  → BatchNorm2d(out_ch)
  → ReLU(inplace=True)
Output: (B, out_ch, H/stride, W/stride)
```

No residual connection. No bias terms (BatchNorm absorbs the bias). Two convs per block = 2× the receptive field growth per block.

### SetEncoder(in_ch=1, feat_dim=64)

Shared-weight encoder applied independently to each SAR track/pol map. Same weights for all maps — the model learns a generic "backscatter change → feature" mapping.

```
Input: (B, 1, 128, 128) — one SAR change map

ConvBlock(1, 16, stride=2)     → (B, 16, 64, 64)
ConvBlock(16, 32, stride=2)    → (B, 32, 32, 32)
ConvBlock(32, 64, stride=2)    → (B, 64, 16, 16)
ConvBlock(64, 64, stride=2)    → (B, 64, 8, 8)

Output: (B, 64, 8, 8)
```

Total downsampling: 128 → 8 (16× reduction). Each output spatial position has a receptive field covering a ~16×16 pixel region of the input.

### SpatialSetAttention(feat_dim=64)

Per-position attention pooling. This is what makes the model handle variable numbers of SAR maps.

```
Inputs: N tensors of (B, 64, 8, 8) — one per track/pol

1. Stack:     (B, N, 64, 8, 8)
2. Reshape:   (B, 64, N, 64)   ← (B, H*W, N, C)
3. Learnable query: (64,) — single vector, learned end-to-end

4. Attention scores:
   score[b, pos, n] = (feature[b, pos, n, :] · query[:]) / sqrt(64)
   → (B, 64, N) raw scores

5. Softmax over N: weights[b, pos, n] = softmax(scores, dim=N)
   → Each position independently decides how much to attend to each input

6. Weighted sum:
   out[b, pos, :] = Σ_n weights[b, pos, n] × feature[b, pos, n, :]
   → (B, 64, 64) → reshape to (B, 64, 8, 8)

Output: (B, 64, 8, 8) — aggregated SAR features
```

Key insight: attention is **per-position**, not global. Each 8x8 spatial location independently weighs the N input maps. This lets the model, for example, attend more to descending track data in some areas and ascending in others.

### StaticEncoder(in_ch=8)

Smaller encoder for the 8 static terrain channels.

```
Input: (B, 8, 128, 128)

ConvBlock(8, 16, stride=4)     → (B, 16, 32, 32)
ConvBlock(16, 24, stride=2)    → (B, 24, 16, 16)
ConvBlock(24, 32, stride=2)    → (B, 32, 8, 8)

Output: (B, 32, 8, 8)
```

Fewer stages (3 vs 4) and smaller channel dims than SetEncoder. The first stride=4 saves compute. Output is 32 channels vs 64 for SAR — terrain gets less weight.

### Fusion

Simple concatenation + projection:

```
cat(sar_feat, static_feat) → (B, 96, 8, 8)   ← 64 + 32
Conv2d(96, 64, 1x1)        → (B, 64, 8, 8)    ← learned mixing
ReLU
```

1x1 convolution lets the model learn which SAR-terrain feature combinations matter.

### Decoder

4-stage upsampling from 8x8 → 128x128:

```
Input: (B, 64, 8, 8)

_DeconvBlock(64, 32):   8  → 16    ConvTranspose(64→32, 4x4, stride=2) + BN-ReLU-Conv-BN-ReLU
_DeconvBlock(32, 16):   16 → 32
_DeconvBlock(16, 8):    32 → 64
_DeconvBlock(8, 4):     64 → 128

Conv2d(4, 1, 1x1)       → (B, 1, 128, 128)  ← logits

Output: (B, 1, 128, 128) raw logits
```

Each `_DeconvBlock`:
```
ConvTranspose2d(in_ch, out_ch, 4x4, stride=2, padding=1, bias=False)  ← learned 2× upsample
BatchNorm2d(out_ch)
ReLU
Conv2d(out_ch, out_ch, 3x3, padding=1, bias=False)  ← refine/sharpen
BatchNorm2d(out_ch)
ReLU
```

The refinement conv after each transpose is important — ConvTranspose alone tends to produce checkerboard artifacts. The 3x3 conv smooths these out.

### Parameter count

~236,789 total parameters. Breakdown:
- SetEncoder: ~4 ConvBlocks ≈ majority of params
- StaticEncoder: 3 ConvBlocks, smaller channels
- SpatialSetAttention: just 64 parameters (the query vector)
- Fusion: 96×64 + 64 = 6,208 (1x1 conv)
- Decoder: 4 DeconvBlocks, shrinking channels

---

## Data Format Reference

### SAR channels (variable count)

Each SAR map is a single-channel (1, 128, 128) backscatter change map.

- Source: `d_{track_id}_{pol}_empirical` variables from `calculate_empirical_backscatter_probability`
- Values: empirical distance / change metric (not probabilities)
- Typical range: roughly [-3, 3] dB-scale
- NaNs replaced with 0.0
- Number varies per scene: typically 2-8 (ascending/descending tracks × VV/VH polarizations)

### Static channels (fixed 8)

Always 8 channels in this order (defined in `channels.py`):

| Index | Channel | Normalization | Range (post-norm) | Description |
|-------|---------|---------------|--------------------|-------------|
| 0 | `fcf` | ÷ 100 | [0, 1] | Forest cover fraction |
| 1 | `slope` | ÷ 0.6 | [0, ~1.3] | Slope angle in radians |
| 2 | `cell_counts` | sign × log1p(abs), ÷ 5 | [0, ~2] | FlowPy runout cell counts |
| 3 | `release_zones` | none | {0, 1} | Binary release zone mask |
| 4 | `runout_angle` | ÷ π | [0, 1] | FlowPy runout angle |
| 5 | `water_mask` | none | {0, 1} | Binary water mask |
| 6 | `d_empirical` | ÷ 5 | [-1, 1] | Combined empirical distance |
| 7 | `dem` | per-patch min-max | [0, 1] | DEM elevation |

DEM is special: left raw during `build_static_stack()`, then per-patch min-max normalized after slicing. This preserves relative elevation within each patch rather than absolute elevation.

### Label format

Two types:
- **Window-level**: single int8 (0 or 1), broadcast to all 128×128 pixels
- **Pixel-level**: (128, 128) float32 mask from rasterized polygons (preferred — from `extract_patches_from_polygons.py`)

### .npz file contents

```python
{
    'sar_maps':   np.float32,  # (N, 128, 128) — variable N
    'static':     np.float32,  # (8, 128, 128) — normalized
    'label':      np.int8,     # 0 or 1
    'label_mask': np.float32,  # (128, 128) — optional pixel-level mask
    'x_coords':   np.float64,  # (128,) — x coordinate values
    'y_coords':   np.float64,  # (128,) — y coordinate values
    'crs':        str,          # CRS string
}
```

---

## Key Constants & Hyperparameters

| Parameter | Value | Where |
|-----------|-------|-------|
| `V2_PATCH_SIZE` | 128 | `patch_extraction.py` |
| `N_STATIC` | 8 | `channels.py` |
| `feat_dim` | 64 | `model.py` (SetEncoder output / Decoder input) |
| StaticEncoder output | 32 channels | `model.py` |
| Fusion | 96 → 64 (1x1 conv) | `model.py` |
| Augmentation noise σ | 0.05 | `dataset.py` |
| Default tau | 6 days | temporal decay for empirical computation |
| Pos weight cap | 50.0 | `train.py` (auto-compute cap) |
| Val IoU threshold | 0.5 | `train.py` |
| Inference stride | 32 | 75% overlap between 128px windows |
| Extraction stride | 64 | 50% overlap for training patches |
| Min debris fraction | 0.005 | 0.5% of patch = ~82 pixels for positive class |
| Neg:pos ratio | 3.0 | Subsample negatives to 3:1 |
| Model parameters | ~236,789 | Verified empirically |
