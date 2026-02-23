import numpy as np
import xarray as xr
import torch
from sarvalanche.ml.inference import load_model, prep_dataset_for_inference, predict_with_sweeping_fast
from sarvalanche.utils.constants import eps

def mahalanobis_distance(
    da: xr.DataArray,
    avalanche_date,
    model: torch.nn.Module,
    device: str = 'mps',
    stride: int = 4,
    batch_size: int = 128,
    max_timesteps: int = 15,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute signed z-score distances for first timestep after avalanche_date.

    For the post-event timestep t, all timesteps strictly before t are used
    as context for the transformer, and t itself is the observation:

        distance[t] = (obs[t] - mu[t]) / sigma[t]

    Positive values → obs higher than predicted (wet snow, rain crust)
    Negative values → obs lower than predicted (dry new snow, deep slab)

    Parameters
    ----------
    da : xr.DataArray
        Time series with dims (time, y, x) or (time, polarization, y, x),
        values in dB. Must have a 'time' coordinate.
    avalanche_date : str
        ISO date string (e.g. '2021-01-15'). Timesteps on or after this
        date are treated as observations; all prior timesteps are context.
    model : torch.nn.Module, optional
        Loaded SARTransformer in eval mode. If None, loads from weights_path
        or the package default weights.
    device : str
        Torch device ('mps', 'cuda', 'cpu').
    stride : int
        Inference stride for the sweeping window (4 = 75% overlap).
    batch_size : int
        Batch size for inference.

    Returns
    -------
    distance : xr.DataArray
        Signed z-score per post-event timestep, dims (time, y, x).
    sigma : xr.DataArray
        Predicted uncertainty per post-event timestep, dims (time, y, x).
    """
    # prep_dataset_for_inference expects (VV, VH) but we pass the same
    # DataArray twice if only one polarization is available — the model
    # was trained on both, so this is a graceful fallback.
    # If da already contains both pols (dim 'polarization'), split them.
    model.eval()

    # Split VV/VH
    if 'polarization' in da.dims:
        vv = da.sel(polarization='VV') if 'VV' in da.polarization.values else da.isel(polarization=0)
        vh = da.sel(polarization='VH') if 'VH' in da.polarization.values else da.isel(polarization=1)
    else:
        vv = da
        vh = da

    # Split timeline at avalanche_date
    event_date = np.datetime64(avalanche_date)
    times      = da.time.values

    pre_mask  = times < event_date
    post_mask = times >= event_date

    post_times = times[post_mask]

    if not pre_mask.any():
        raise ValueError(f'No timesteps before {avalanche_date} to use as context.')
    if not post_mask.any():
        raise ValueError(f'No timesteps on or after {avalanche_date} to evaluate.')

    t = post_times[0]
    with torch.no_grad():
        # for t in post_times:

        vv_ctx = vv.isel(time=pre_mask).isel(time=slice(-max_timesteps, None))
        vh_ctx = vh.isel(time=pre_mask).isel(time=slice(-max_timesteps, None))
        vv_obs = vv.sel(time=t)
        vh_obs = vh.sel(time=t)  # not used in distance but kept for consistency

        # Stack context + observation as the last timestep
        vv_seq = xr.concat([vv_ctx, vv_obs.expand_dims('time')], dim='time')
        vh_seq = xr.concat([vh_ctx, vh_obs.expand_dims('time')], dim='time')

        data = prep_dataset_for_inference(vv_seq, vh_seq)  # (T, 2, H, W)

        mu_np, sigma_np = predict_with_sweeping_fast(
            model, data[:-1],
            stride=stride,
            batch_size=batch_size,
            use_fp16=False,
            min_valid_fraction=0.01,
            device=device,
        )

    mu_np    = np.asarray(mu_np)
    sigma_np = np.asarray(sigma_np)
    if mu_np.ndim    == 3: mu_np    = mu_np[0]
    if sigma_np.ndim == 3: sigma_np = sigma_np[0]

    obs_np  = np.asarray(vv_obs)   # (H, W)
    dist_np = (obs_np - mu_np) / (sigma_np + eps)

    y_coords = vv.coords['y'] if 'y' in vv.coords else np.arange(obs_np.shape[0])
    x_coords = vv.coords['x'] if 'x' in vv.coords else np.arange(obs_np.shape[1])
    coords   = {'y': y_coords, 'x': x_coords}

    distance = xr.DataArray(dist_np,  dims=['y', 'x'], coords=coords)
    sigma = xr.DataArray(sigma_np, dims=['y', 'x'], coords=coords)

    distance.name = 'ml_distance'
    sigma.name    = 'ml_sigma'

    distance.attrs = {
        'units':          'standard_deviations',
        'source':         'sarvalanche',
        'product':        'ml_transformer_distance',
        'avalanche_date': avalanche_date,
    }
    sigma.attrs = {
        'units':          'dB',
        'source':         'sarvalanche',
        'product':        'ml_transformer_sigma',
        'avalanche_date': avalanche_date,
    }

    return distance, sigma
