
import logging

import torch
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt

from sarvalanche.ml.SARTransformer import SARTransformer

log = logging.getLogger(__name__)

def load_model(pth_filepath):
    checkpoint = torch.load(pth_filepath, map_location='cpu')

    # If model_config exists:
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = SARTransformer(**config)
    else:
        # Manually match what you used during training
        # Based on the error, it looks like you trained with in_chans=2
        model = SARTransformer(
            img_size=16,
            patch_size=8,
            in_chans=2,
            embed_dim=256,
            depth=4,
            num_heads=4,
        )

    # 3. Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Set to evaluation mode
    # model.eval()

    return model

def prep_dataset_for_inference(VV, VH):
    combined = xr.concat([VV, VH], dim='polarization')
    combined = combined.assign_coords(polarization=['VV', 'VH'])
    combined = combined.transpose('time', 'polarization', 'y', 'x')
    # model only trained for at most last 10 time steps
    combined = combined.isel(time=slice(-10, None))

    return combined

def predict_with_sweeping_fast(model, baseline, patch_size=16, stride=8,
                               batch_size=128, device=None, use_fp16=False,
                               min_valid_fraction=0.5):
    """
    Fast batched inference with configurable options.

    Recommended settings for speed:
    - stride=8 (good quality/speed balance)
    - batch_size=128 (for GPU) or 64 (for CPU)
    - use_fp16=True (if GPU supports it)
    - min_valid_fraction=0.5 (skip patches with <50% valid pixels)
    """
    model.eval()

    # Device setup
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            log.info("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            log.info("Using CUDA GPU")
        else:
            device = torch.device('cpu')
            log.info("Using CPU")
    else:
        device = torch.device(device)

    model = model.to(device)

    if use_fp16 and device.type in ['cuda', 'mps']:
        model = model.half()

    # Prepare data
    if isinstance(baseline, xr.DataArray):
        baseline_np = baseline.values
    else:
        baseline_np = baseline

    if baseline_np.ndim == 3:
        T, H, W = baseline_np.shape
        C = 1
        baseline_np = baseline_np[:, None, :, :]
    else:
        T, C, H, W = baseline_np.shape

    # Create validity mask - pixels valid across ALL time and channels
    valid_mask = np.all(np.isfinite(baseline_np), axis=(0, 1))  # (H, W)
    log.debug(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")

    # Fill NaNs with mean of valid data (model can't process NaN)
    if not np.all(valid_mask):
        fill_value = np.nanmean(baseline_np[np.isfinite(baseline_np)])
        baseline_filled = baseline_np.copy()
        for t in range(T):
            for c in range(C):
                baseline_filled[t, c, ~valid_mask] = fill_value
    else:
        baseline_filled = baseline_np

    # Extract patches, skipping invalid ones
    patches = []
    positions = []
    total_patches = 0
    skipped_patches = 0

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            total_patches += 1

            # Check if patch has enough valid pixels
            patch_valid_mask = valid_mask[i:i+patch_size, j:j+patch_size]
            valid_fraction = patch_valid_mask.sum() / (patch_size * patch_size)

            if valid_fraction < min_valid_fraction:
                skipped_patches += 1
                continue  # Skip this patch

            # Add patch to processing list
            patches.append(baseline_filled[:, :, i:i+patch_size, j:j+patch_size])
            positions.append((i, j))

    log.debug(f"Processing {len(patches)}/{total_patches} patches (skipped {skipped_patches}, {100*skipped_patches/total_patches:.1f}%)")

    patches = np.array(patches)

    # Initialize outputs
    mu_sum = np.zeros((C, H, W), dtype=np.float32)
    sigma_sum = np.zeros((C, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    # Batched inference
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(patches), batch_size), desc='Processing batches'):
            batch_end = min(batch_start + batch_size, len(patches))

            batch_patches = torch.FloatTensor(patches[batch_start:batch_end])
            if use_fp16:
                batch_patches = batch_patches.half()
            batch_patches = batch_patches.to(device)

            mu_batch, sigma_batch = model(batch_patches)

            mu_batch = mu_batch.cpu().float().numpy()
            sigma_batch = sigma_batch.cpu().float().numpy()

            for idx, (i, j) in enumerate(positions[batch_start:batch_end]):
                mu_sum[:, i:i+patch_size, j:j+patch_size] += mu_batch[idx]
                sigma_sum[:, i:i+patch_size, j:j+patch_size] += sigma_batch[idx]
                count[i:i+patch_size, j:j+patch_size] += 1

    # Average by dividing by count
    mu = mu_sum / (count[None, :, :] + 1e-8)
    sigma = sigma_sum / (count[None, :, :] + 1e-8)

    # Restore NaNs where original data was invalid
    mu[:, ~valid_mask] = np.nan
    sigma[:, ~valid_mask] = np.nan

    # Handle valid pixels that got no predictions (edges near invalid areas)
    no_prediction = (count == 0) & valid_mask
    if no_prediction.any():
        log.debug(f"{no_prediction.sum()} valid edge or {100*no_prediction.sum()/mu.size:.1f}% pixels had no predictions, filling...")
        if no_prediction.sum()/mu.size > 0.2: log.warning(f"{no_prediction.sum()} valid edge or {100*no_prediction.sum()/mu.size:.1f}% pixels had no predictions, filling...")
        for c in range(C):
            if no_prediction.any():
                indices = distance_transform_edt(no_prediction, return_distances=False, return_indices=True)
                mu[c][no_prediction] = mu[c][tuple(indices[:, no_prediction])]
                sigma[c][no_prediction] = sigma[c][tuple(indices[:, no_prediction])]

    if isinstance(baseline, xr.DataArray):
        mu = xr.DataArray(mu, dims=['polarization', 'y', 'x'], coords={'polarization': baseline.polarization, 'y': baseline.y, 'x': baseline.x})
        sigma = xr.DataArray(sigma, dims=['polarization', 'y', 'x'], coords={'polarization': baseline.polarization, 'y': baseline.y, 'x': baseline.x})

    return mu, sigma

def compute_mahalanobis_with_sweeping(model, da, avalanche_date, stride=4, device='cpu'):
    """
    Compute Mahalanobis distance on full scene using sweeping inference.

    Parameters
    ----------
    model : SARTransformer
        Trained model
    da : xr.DataArray
        SAR backscatter timeseries (time, y, x)
    avalanche_date : str
        Date of avalanche event
    stride : int
        Stride for sweeping (4 = paper's recommendation for speed/quality tradeoff)
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    distance : xr.DataArray
        Mahalanobis distance for first post-event image
    """
    # Split data
    prior = da.sel(time=slice(None, avalanche_date))
    post = da.sel(time=slice(avalanche_date, None))

    log.info("Using %d baseline images", len(prior.time))
    log.debug("Image size: %d x %d", prior.shape[-2], prior.shape[-1])

    # Predict distribution using sweeping
    mu, sigma = predict_with_sweeping_fast(
        model,
        prior,
        patch_size=16,
        stride=stride,
        device=device
    )

    # Get first post-event observation
    actual = post.isel(time=0).values

    # Apply same transform as training (logit)
    actual = np.log(actual / (1 - actual))

    # Compute distance
    if mu.shape[0] == 1:  # Single channel
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)

    distance = np.abs(actual - mu) / (sigma + 1e-8)

    # Convert to DataArray
    distance_da = xr.DataArray(
        distance,
        dims=['y', 'x'],
        coords={'y': post.y, 'x': post.x, 'time': post.time[0]}
    )

    distance_da.attrs = {
        'source': 'sarvalanche',
        'units': 'standard_deviations',
        'product': 'mahalanobis_distance',
        'method': 'transformer_sweeping',
        'stride': stride
    }

    return distance_da

"""
Unified inference pipeline for avalanche debris detection.

Combines three models in sequence:
  1. SAR Transformer  — pre-computed upstream, produces unmasked_p_target in ds
  2. XGBoost          — scores all tracks cheaply, produces p_debris per track
  3. CNN Segmenter    — runs only on tracks where p_debris >= cnn_threshold,
                        produces pixel-wise debris probability mask per track

Usage
-----
    from sarvalanche.ml.inference import DebrisInference

    pipeline = DebrisInference.load()
    result_gdf = pipeline.predict(gdf, ds, src_crs=gdf.crs)

    # result_gdf has columns:
    #   p_debris  : float in [0, 1] — XGBoost debris probability
    #   seg_mask  : np.ndarray (64, 64) or None — CNN segmentation mask
    #               None for tracks below cnn_threshold
"""

import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import torch
import xarray as xr
from tqdm.auto import tqdm

from sarvalanche.ml.debris_segmenter import DebrisSegmenter
from sarvalanche.ml.track_classifier import (
    STATIC_FEATURE_VARS,
    predict_tracks,
)
from sarvalanche.ml.track_features import (
    compute_scene_context,
    precompute_group_arrays,
)
from sarvalanche.ml.track_patch_extraction import extract_context_patch
from sarvalanche.ml.weight_utils import find_weights
from sarvalanche.ml.SARTransformer import SARTransformer
# from sarvalanche.ml.rtc_inference import load_model, prep_dataset_for_inference, predict_with_sweeping_fast

log = logging.getLogger(__name__)

# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Main inference class ──────────────────────────────────────────────────────

class DebrisInference:
    """Unified debris detection pipeline.

    Parameters
    ----------
    xgb_clf : XGBClassifier
        Fitted XGBoost track classifier.
    seg_model : DebrisSegmenter
        Fitted CNN segmentation model.
    sar_transformer : SARTransformer or None
        SAR transformer for upstream RTC prediction. If None, assumes
        unmasked_p_target is already present in the dataset.
    cnn_threshold : float
        XGBoost p_debris threshold above which the CNN is run.
        Tracks below this threshold get seg_mask=None.
    patch_size : int
        Patch size for CNN input. Must match training size.
    device : torch.device
        Device for CNN inference.
    """

    def __init__(
        self,
        xgb_clf,
        seg_model: DebrisSegmenter,
        cnn_threshold: float = 0.5,
        patch_size: int = 64,
        device: torch.device | None = None,
    ):
        self.xgb_clf        = xgb_clf
        self.seg_model      = seg_model
        self.cnn_threshold  = cnn_threshold
        self.patch_size     = patch_size
        self.device         = _resolve_device(device)

        self.seg_model.eval()
        self.seg_model.to(self.device)

        log.info(
            'DebrisInference ready — device=%s, cnn_threshold=%.2f',
            self.device, self.cnn_threshold,
        )

    @classmethod
    def load(
        cls,
        cnn_threshold: float = 0.5,
        patch_size: int = 64,
        device: str | torch.device | None = None,
        load_sar_transformer: bool = False,
    ) -> 'DebrisInference':
        """Load all models from the weights registry.

        Parameters
        ----------
        cnn_threshold : float
            XGBoost gate threshold for CNN.
        patch_size : int
            Must match the patch size used during CNN training.
        device : str or torch.device or None
            Force a specific device. Auto-detected if None.
        load_sar_transformer : bool
            Load the SAR transformer model. Set False if unmasked_p_target
            is already present in the dataset (the common case).

        Returns
        -------
        DebrisInference
        """
        resolved_device = _resolve_device(device)

        # XGBoost
        xgb_path = find_weights('track_classifier')
        xgb_clf  = joblib.load(xgb_path)
        log.info('Loaded XGBoost from %s (%d features)', xgb_path, len(xgb_clf.feature_names_in_))

        # CNN segmenter
        seg_path  = find_weights('debris_segmenter')
        seg_model = DebrisSegmenter(patch_size=patch_size)
        seg_model.load_state_dict(
            torch.load(seg_path, map_location=resolved_device)
        )
        log.info('Loaded CNN segmenter from %s', seg_path)


        return cls(
            xgb_clf=xgb_clf,
            seg_model=seg_model,
            cnn_threshold=cnn_threshold,
            patch_size=patch_size,
            device=resolved_device,
        )

    def predict(
        self,
        gdf: gpd.GeoDataFrame,
        ds: xr.Dataset,
        src_crs=None,
        n_jobs: int = 1,
        show_progress: bool = True,
        prior_estimate=None,
    ) -> gpd.GeoDataFrame:
        """Run the full inference pipeline on all tracks.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Track polygons.
        ds : xr.Dataset
            Dataset in WGS84 with SAR and terrain variables.
            Must contain unmasked_p_target unless sar_transformer is loaded.
        src_crs : CRS or None
            CRS of gdf.geometry. Defaults to gdf.crs.
        n_jobs : int
            Parallel workers for XGBoost feature extraction.
        show_progress : bool
            Show tqdm progress bar for CNN inference.
        prior_estimate : float or array-like or None
            Prior debris probability to shift XGBoost outputs via log-odds.
            0.5 = neutral (no change), >0.5 pushes up, <0.5 pushes down.
            None = no adjustment (default).

        Returns
        -------
        gpd.GeoDataFrame
            Copy of gdf with added columns:
              p_debris : float — XGBoost debris probability
              seg_mask : object — (patch_size, patch_size) float32 ndarray
                         or None for tracks below cnn_threshold
        """
        src_crs = src_crs or gdf.crs
        result  = gdf.copy()

        # ── Step 1: XGBoost — score all tracks ───────────────────────────────
        log.info('Step 1/2: XGBoost scoring %d tracks...', len(gdf))
        result = predict_tracks(self.xgb_clf, gdf, ds, n_jobs=n_jobs, prior_estimate=prior_estimate)

        # ── Step 2: CNN — score tracks above threshold ────────────────────────
        cnn_candidates = result[result['p_debris'] >= self.cnn_threshold]
        log.info(
            'Step 2/2: CNN segmentation on %d / %d tracks (p_debris >= %.2f)',
            len(cnn_candidates), len(gdf), self.cnn_threshold,
        )

        seg_masks = {idx: None for idx in gdf.index}

        if len(cnn_candidates) > 0:
            iterator = tqdm(
                cnn_candidates.iterrows(),
                total=len(cnn_candidates),
                desc='CNN segmentation',
                unit='track',
                disable=not show_progress,
            )
            with torch.no_grad():
                for idx, row in iterator:
                    try:
                        patch = extract_context_patch(
                            row, ds,
                            size=self.patch_size,
                            src_crs=src_crs,
                        )
                        x      = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
                        logits = self.seg_model(x)
                        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
                        seg_masks[idx] = probs.astype(np.float32)
                    except Exception as exc:
                        log.warning('CNN failed on track %s: %s', idx, exc)
                        seg_masks[idx] = None

        result['seg_mask'] = [seg_masks[idx] for idx in result.index]

        n_segmented = sum(1 for m in result['seg_mask'] if m is not None)
        log.info(
            'predict: %d tracks scored, %d segmented, mean p_debris=%.3f',
            len(result), n_segmented, float(result['p_debris'].mean()),
        )
        return result

    def predict_single(
        self,
        row: gpd.GeoSeries,
        ds: xr.Dataset,
        src_crs=None,
        scene_ctx: dict | None = None,
        prior_estimate=None,
    ) -> tuple[float, np.ndarray | None]:
        """Score a single track — useful for debugging and notebooks.

        Parameters
        ----------
        row : gpd.GeoSeries
        ds : xr.Dataset
        src_crs : CRS or None
        scene_ctx : dict or None
            Pre-computed scene context. Computed fresh if None.
        prior_estimate : float or None
            Prior debris probability. 0.5 = neutral, >0.5 pushes up.
            None = no adjustment.

        Returns
        -------
        p_debris : float
        seg_mask : np.ndarray (patch_size, patch_size) or None
        """
        from sarvalanche.ml.track_features import extract_track_features
        from sarvalanche.ml.track_classifier import _apply_prior

        if scene_ctx is None:
            scene_ctx = compute_scene_context(ds)

        feats = extract_track_features(
            row, ds, scene_ctx=scene_ctx, src_crs=src_crs or row.crs
        )

        import pandas as pd
        X = pd.DataFrame([feats]).reindex(
            columns=self.xgb_clf.feature_names_in_
        ).fillna(0)
        p_debris = float(self.xgb_clf.predict_proba(X)[0, 1])

        if prior_estimate is not None:
            p_debris = float(_apply_prior(np.array([p_debris]), prior_estimate)[0])

        seg_mask = None
        if p_debris >= self.cnn_threshold:
            try:
                patch = extract_context_patch(
                    row, ds, size=self.patch_size, src_crs=src_crs
                )
                x = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.seg_model(x))
                seg_mask = probs.squeeze().cpu().numpy().astype(np.float32)
            except Exception as exc:
                log.warning('CNN failed on predict_single: %s', exc)

        return p_debris, seg_mask