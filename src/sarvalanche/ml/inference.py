
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
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")
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
    log.info(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")

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

    log.info(f"Processing {len(patches)}/{total_patches} patches (skipped {skipped_patches}, {100*skipped_patches/total_patches:.1f}%)")

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
        log.warning(f"{no_prediction.sum()} valid edge or {100*no_prediction.sum()/mu.size:.1f}% pixels had no predictions, filling...")
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

    print(f"Using {len(prior.time)} baseline images")
    print(f"Image size: {prior.shape[-2]} x {prior.shape[-1]}")

    # Predict distribution using sweeping
    mu, sigma = predict_with_sweeping(
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
