
from itertools import product

import numpy as np
import xarray as xr

from scipy.stats import norm

from sarvalanche.utils.constants import pols, eps

def ecdf_survival_1pixel(ref_vals, post_vals, min_ref):
    """
    ref_vals:  (T_ref,)
    post_vals: (T_post,)
    returns:   (T_post,)
    """
    ref_vals = ref_vals[~np.isnan(ref_vals)]

    out = np.full(post_vals.shape, np.nan, dtype=np.float32)

    if ref_vals.size < min_ref:
        return out

    for i, x in enumerate(post_vals):
        if np.isnan(x):
            continue
        out[i] = np.mean(ref_vals >= x)

    return out

def channel_pvals_ufunc(
    da_ref: xr.DataArray,
    da_post: xr.DataArray,
    min_ref: int,
):
    """
    da_ref:  (time_ref, y, x)
    da_post: (time_post, y, x)
    returns: (time_post, y, x)
    """

    time_post_len = da_post.sizes["time"]
    p = xr.apply_ufunc(
        ecdf_survival_1pixel,
        da_ref.rename({"time": "time_ref"}),
        da_post.rename({"time": "time_post"}),
        kwargs={"min_ref": min_ref},
        input_core_dims=[["time_ref"], ["time_post"]],
        output_core_dims=[["time_post"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        output_sizes={"time_post": time_post_len}
    )

    p = p.transpose("time_post", "y", "x")
    p = p.rename({'time_post': 'time'})
    return p

def compute_channel_stats(ds, pol, track, event_date, min_ref=4, n_ref=15):
    """
    Compute per-channel, per-track stats: pvals, sigma, sign, lia
    """
    # Select channel
    da_chan = ds[pol].sel(time=ds.track == track)

    # Split time
    da_pre = da_chan.sel(time=slice(None, event_date))
    da_post = da_chan.sel(time=slice(event_date, None))

    if da_pre.time.size < min_ref:
        return None  # skip this channel

    da_ref = da_pre.isel(time=slice(-n_ref, None))

    # --- Apply your per-channel pval function ---
    p = channel_pvals_ufunc(
        da_ref=da_ref,
        da_post=da_post,
        min_ref=min_ref,
    )
    channel_name = f"{pol}_{track}"
    p = p.expand_dims(channel=[channel_name])

    # Pre-event std
    sigma = da_ref.std('time').expand_dims(channel=[channel_name])

    # Sign of change
    median = da_ref.median('time')
    sign = xr.apply_ufunc(
        np.sign,
        da_post - median,
        dask="parallelized",
        output_dtypes=[np.int8],
    ).astype(np.int8).drop(['direction', 'platform', 'track'])
    sign = sign.expand_dims(channel=[channel_name])

    # Local incidence angle
    lia = ds['lia'].sel(static_track=track).expand_dims(channel=[channel_name])

    return p, sigma, sign, lia

def get_orbit_pol_ecdf_pvals(ds, event_date="2020-01-11", min_ref=4, n_ref=15):
    pvals, sigmas, diffs, lias = [], [], [], []

    for pol, track in product(pols, np.unique(ds.track.values)):
        res = compute_channel_stats(ds, pol, track, event_date, min_ref, n_ref)
        if res is None:
            continue
        p, sigma, sign, lia = res
        pvals.append(p)
        sigmas.append(sigma)
        diffs.append(sign)
        lias.append(lia)

    # Concatenate and stack
    p_channel = xr.concat(pvals, dim="channel").stack(obs=("channel", "time")).dropna("obs", how='all')
    signs = xr.concat(diffs, dim="channel").stack(obs=("channel", "time")).dropna("obs", how='all')

    sigmas = xr.concat(sigmas, dim="channel").stack(obs_channel=('channel',)).dropna('obs_channel', how='all')
    sigmas['obs_channel'] = [i[0] for i in sigmas.obs_channel.values]

    lias   = xr.concat(lias, dim="channel").stack(obs_channel=('channel',)).dropna('obs_channel', how='all')
    lias['obs_channel'] = [i[0] for i in lias.obs_channel.values]

    return lias, sigmas, signs, p_channel


def generate_z_signed(p_channel, signs):
    p_clip = xr.where(
        p_channel < eps, eps,
        xr.where(p_channel > 1 - eps, 1 - eps, p_channel)
    ).astype(float)

    z = xr.apply_ufunc(
        norm.ppf,
        1.0 - p_clip,
        dask="parallelized",
        output_dtypes=[float],
    )

    z_signed = z * signs

    return z_signed

def generate_weights(
        sigmas,
        lias,
        z_signed,
        q_pol = {'VV': 1.0, 'VH': 0.8}
):
    pol = xr.DataArray(
    [c.split("_")[0] for c in sigmas.obs_channel.values],
    dims="obs_channel"
    )

    q_pol_da = xr.apply_ufunc(
        lambda p: q_pol[p],
        pol,
        vectorize=True,
        output_dtypes=[float],
    )

    alpha = 1.0

    weights = (
        (1.0 / sigmas)
        * np.cos(np.deg2rad(lias)) ** alpha
        * q_pol_da
    )
    obs_index = z_signed.indexes["obs"]
    channels = obs_index.get_level_values("channel")
    weights = weights.sel(obs_channel=channels)
    weights = weights.rename(obs_channel="obs")
    weights = weights.assign_coords(obs=z_signed.obs)

    return weights

def get_z_norm(z_signed, weights):
    valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)

    w_eff = weights * valid

    num = (w_eff * z_signed).sum("obs", skipna=True)
    den = np.sqrt((w_eff ** 2).sum("obs", skipna=True))

    Z_combined = num / den

    num_obs = valid.sum("obs")
    Z_norm = (Z_combined / num_obs)

    return Z_norm

def z_norm_to_probability(Z_norm, beta = 10.0, Z_pivot = 0.5):
    P_change = 1 / (1 + np.exp(-beta * (Z_norm - Z_pivot)))

    return P_change

def calculate_ecdf_backscatter_probability(ds, event_date):
    lias, sigmas, signs, p_channel = get_orbit_pol_ecdf_pvals(ds, event_date)
    z_signed = generate_z_signed(p_channel, signs)
    weights = generate_weights(sigmas, lias, z_signed, q_pol={'VV':1.0, 'VH':0.8})
    z_norm = get_z_norm(z_signed, weights)
    change_proability = z_norm_to_probability(z_norm, beta = 10, Z_pivot=0.5)
    return change_proability

