
import torch
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt

def predict_with_sweeping(model, baseline, patch_size=16, stride=4, device='cpu'):
    """
    Sweep model across large image with overlapping windows and average predictions.

    This reduces edge artifacts by averaging predictions from overlapping patches.

    Parameters
    ----------
    model : SARTransformer
        Trained model
    baseline : xr.DataArray or np.ndarray
        Baseline images (T, H, W) or (T, C, H, W)
    patch_size : int
        Size of patches (default: 16)
    stride : int
        Stride for sweeping (default: 4). Smaller = more overlap = smoother but slower
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    mu : np.ndarray
        Predicted mean (C, H, W)
    sigma : np.ndarray
        Predicted std (C, H, W)
    """
    model.eval()

    # Convert to numpy if needed
    if isinstance(baseline, xr.DataArray):
        baseline_np = baseline.values
    else:
        baseline_np = baseline

    # Get dimensions
    if baseline_np.ndim == 3:  # (T, H, W)
        T, H, W = baseline_np.shape
        C = 1
        baseline_np = baseline_np[:, None, :, :]  # Add channel dim
    else:  # (T, C, H, W)
        T, C, H, W = baseline_np.shape

    # Initialize accumulators for averaging
    mu_sum = np.zeros((C, H, W), dtype=np.float32)
    sigma_sum = np.zeros((C, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)  # Track how many predictions per pixel

    # Sweep across image
    with torch.no_grad():
        for i in tqdm(range(0, H - patch_size + 1, stride), desc='Sweeping rows'):
            for j in range(0, W - patch_size + 1, stride):
                # Extract patch
                patch = baseline_np[:, :, i:i+patch_size, j:j+patch_size]  # (T, C, 16, 16)

                # Convert to tensor and add batch dim
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)  # (1, T, C, 16, 16)

                # Predict
                mu_patch, sigma_patch = model(patch_tensor)

                # Convert to numpy
                mu_patch = mu_patch.squeeze(0).cpu().numpy()  # (C, 16, 16)
                sigma_patch = sigma_patch.squeeze(0).cpu().numpy()  # (C, 16, 16)

                # Accumulate
                mu_sum[:, i:i+patch_size, j:j+patch_size] += mu_patch
                sigma_sum[:, i:i+patch_size, j:j+patch_size] += sigma_patch
                count[i:i+patch_size, j:j+patch_size] += 1

    # Average by dividing by count
    mu = mu_sum / (count[None, :, :] + 1e-8)
    sigma = sigma_sum / (count[None, :, :] + 1e-8)

    # Handle edges that weren't covered (fill with nearest valid prediction)
    mask = count == 0
    if mask.any():
        print(f"Warning: {mask.sum()} pixels had no predictions (edges). Using nearest neighbor fill.")

        for c in range(C):
            # Find nearest valid pixel for uncovered areas
            valid_mask = ~mask
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            mu[c][mask] = mu[c][tuple(indices[:, mask])]
            sigma[c][mask] = sigma[c][tuple(indices[:, mask])]

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
