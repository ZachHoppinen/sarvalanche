"""
ECDF-based backscatter change detection for avalanche debris.

This module implements avalanche detection based on statistical comparison of
post-event backscatter against the pre-event distribution. For each track/polarization:
1. Convert to dB and optionally smooth
2. Compute ECDF p-values (how unusual is post-event backscatter?)
3. Convert to z-scores with directional sign
4. Weight by pixel stability, viewing geometry, and polarization quality
5. Combine weighted z-scores and convert to probability
6. Fuse across tracks/pols
"""

import logging
import warnings
import numpy as np
import xarray as xr
from scipy.stats import norm

from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.utils.constants import eps
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.weighting import combine_weights

log = logging.getLogger(__name__)

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
    log.debug(f'Time post length :{time_post_len}')
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
        dask_gufunc_kwargs={'output_sizes': {'new_dim': time_post_len}}
    )

    p = p.transpose("time_post", "y", "x")
    p = p.rename({'time_post': 'time'})
    return p

def compute_channel_stats(ds, pol, track, event_date, min_ref=4, n_ref=15):
    """
    Compute per-channel, per-track stats: pvals, sigma, sign, lia
    """
    # Select channel
    log.debug(f'Running computer channel stats on {ds}, {pol}, {track}, {event_date}')
    log.debug(f'Computer channel stats run with: min_ref: {min_ref} and n_ref used: {n_ref}')
    da_chan = ds[pol].sel(time=ds.track == track)

    # Split time
    da_pre = da_chan.sel(time=slice(None, event_date))
    da_post = da_chan.sel(time=slice(event_date, None))
    log.debug(f'Size pre:{da_pre.size}, size post: {da_post.size}')

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
    with warnings.catch_warnings():
        # Check data quality before computing std
        valid_count = da_ref.notnull().sum('time')
        total_pixels = valid_count.size
        insufficient_data = (valid_count <= 1).sum().item()
        sufficient_pct = 100 * (1 - insufficient_data / total_pixels)
        log.debug(f'valid count for standard devation calc: {valid_count} of {total_pixels}')

        # Warn if too many pixels have insufficient data
        if sufficient_pct < 50:
            log.warning(
                f"Channel {channel_name}: Only {sufficient_pct:.1f}% of pixels have sufficient data "
                f"({insufficient_data}/{total_pixels} pixels with ≤1 observations)"
            )

        # some all nan slices in time.
        warnings.filterwarnings('ignore', 'Degrees of freedom <= 0', RuntimeWarning)
        sigma = da_ref.std('time').expand_dims(channel=[channel_name])

    # Sign of change
    log.debug('Getting median backscatter for sign calculation')
    median = da_ref.median('time')
    sign = xr.apply_ufunc(
        np.sign,
        da_post - median,
        dask="parallelized",
        output_dtypes=[np.int8],
    ).drop(['direction', 'platform', 'track'])
    sign = sign.expand_dims(channel=[channel_name])

    # Local incidence angle
    log.debug(f'Local incidence angle selected for weighting.')
    lia = ds['lia'].sel(static_track=track).expand_dims(channel=[channel_name])

    return p, sigma, sign, lia

def get_orbit_pol_ecdf_pvals(ds, event_date="2020-01-11", min_ref=4, n_ref=15):
    pvals, sigmas, diffs, lias = [], [], [], []

    for pol, track in product(pols, np.unique(ds.track.values)):
        res = compute_channel_stats(ds, pol, track, event_date, min_ref, n_ref)
        if res is None:
            log.debug(f'No result genreated from computing channel stats.')
            continue
        p, sigma, sign, lia = res
        pvals.append(p)
        sigmas.append(sigma)
        diffs.append(sign)
        lias.append(lia)

    # Concatenate and stack
    log.debug(f'Concat pols and tracks to single joint channel dimension.')
    p_channel = xr.concat(pvals, dim="channel", join='outer').stack(obs=("channel", "time")).dropna("obs", how='all')
    signs = xr.concat(diffs, dim="channel", join='outer').stack(obs=("channel", "time")).dropna("obs", how='all')

    sigmas = xr.concat(sigmas, dim="channel", join='outer')
    sigmas = sigmas.stack(obs_channel=('channel',)).dropna('obs_channel', how='all')
    sigmas = sigmas.drop_vars(['obs_channel', 'channel']).assign_coords(obs_channel=[i[0] for i in sigmas.obs_channel.values])

    lias = xr.concat(lias, dim="channel", join='outer', coords='different', compat='equals')
    lias = lias.stack(obs_channel=('channel',)).dropna('obs_channel', how='all')
    lias = lias.drop_vars(['obs_channel', 'channel']).assign_coords(obs_channel=[i[0] for i in lias.obs_channel.values])

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
        q_pol = {'VV': 1.0, 'VH': 0.8},
        lia_optimal=55.0,
        lia_width=20.0
):
    """
    Calculate observation weights based on temporal stability, local incidence angle,
    and polarization quality.

    Based on Bühler et al. findings that avalanches appear brightest at
    incidence angles of 55 ± 20°.

    Parameters
    ----------
    sigmas : array-like
        Temporal standard deviation of backscatter (lower = more stable)
    lias : array-like
        Local incidence angles in degrees
    q_pol_da : array-like
        Polarization quality factor
    lia_optimal : float
        Optimal local incidence angle for avalanche detection (default 55°)
    lia_width : float
        Width of optimal angle range (default 25°, giving 30-80° range)

    Returns
    -------
    weights : array-like
        Combined observation weights
    """

    log.debug(f'Using q_pol {q_pol}')

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

    log.debug(f'Using center of best lias as {lia_optimal} with width +- {lia_width}')
    # 2. Local incidence angle weight (Gaussian centered at optimal angle)
    # This heavily weights 35-75° range, with peak at 55°
    lia_weight = incidence_angle_weight(lias, lia_optimal, lia_width)

    pixel_sigma_weighting(sigmas, tau_variability=20.0)

    weights = (
        (1.0 / sigmas)
        *lia_weight
        * q_pol_da
    )
    log.debug(f'Total weights: {weights}')
    obs_index = z_signed.indexes["obs"]
    channels = obs_index.get_level_values("channel")
    weights = weights.sel(obs_channel=channels)
    weights = weights.rename(obs_channel="obs")
    weights = weights.assign_coords(obs=z_signed.obs)

    return weights

def get_z_norm(z_signed, weights):
    valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)
    log.debug(f'Running z_score normalization by number of observations')
    log.debug(f'Valid number for z_score: {valid.size}')

    w_eff = weights * valid

    num = (w_eff * z_signed).sum("obs", skipna=True)
    den = np.sqrt((w_eff ** 2).sum("obs", skipna=True))

    Z_combined = num / den

    num_obs = valid.sum("obs")
    Z_norm = (Z_combined / num_obs)

    return Z_norm

def z_norm_to_probability(Z_norm, beta = 10.0, Z_pivot = 0.5):
    log.debug(f'COnverting z norm to probability using: beta: {beta} and Z_pivot: {Z_pivot}')
    P_change = 1 / (1 + np.exp(-beta * (Z_norm - Z_pivot)))

    return P_change

def calculate_ecdf_backscatter_probability(ds, event_date):
    log.debug('Calculating Distribution based change probability.')
    lias, sigmas, signs, p_channel = get_orbit_pol_ecdf_pvals(ds, event_date)
    z_signed = generate_z_signed(p_channel, signs)
    weights = generate_weights(sigmas, lias, z_signed, q_pol={'VV':1.0, 'VH':0.8})
    z_norm = get_z_norm(z_signed, weights)
    change_proability = z_norm_to_probability(z_norm, beta = 10, Z_pivot=0.5)
    return change_proability


from scipy.stats import norm

from sarvalanche.utils.generators import iter_track_pol_combinations
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.utils.constants import eps
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.features.incidence_angle import incidence_angle_weight
from sarvalanche.features.stability import pixel_sigma_weighting
from sarvalanche.features.weighting import combine_weights
from sarvalanche.detection.probability import _z_to_probability, log_odds_combine

def calculate_ecdf_backscatter_probability(
    ds: xr.Dataset,
    avalanche_date,
    *,
    polarizations=("VV", "VH"),
    smooth_method=None,
    min_ref: int = 4,
    n_ref: int = 15,
    q_pol: dict = {'VV': 1.0, 'VH': 0.8},
    lia_optimal: float = 55.0,
    lia_width: float = 20.0,
    beta: float = 10.0,
    z_pivot: float = 0.5,
    combine_alpha: float = 0.5,
):
    """
    Compute avalanche debris probability using ECDF-based statistical testing.
    
    [docstring same as before...]
    """
    log.info(f"Computing ECDF probabilities for {len(polarizations)} polarizations")

    probability_list: list[xr.DataArray] = []

    for track, pol, da, lia in iter_track_pol_combinations(
        ds,
        polarizations=polarizations,
        include_lia=True,
        skip_missing=True
    ):
        log.debug(f"Processing track={track}, pol={pol}")

        # --- Convert to dB and smooth ---
        if check_db_linear(da) != 'dB':
            da = linear_to_dB(da)

        if smooth_method is not None:
            da = spatial_smooth(da, method=smooth_method)

        # --- Split pre/post event ---
        da_pre = da.sel(time=slice(None, avalanche_date))
        da_post = da.sel(time=slice(avalanche_date, None))

        if da_pre.time.size < min_ref:
            log.warning(f"Skipping {pol}_{track}: only {da_pre.time.size} pre-event observations")
            continue

        # Use most recent n_ref observations as reference
        da_ref = da_pre.isel(time=slice(-n_ref, None))

        # --- Compute ECDF p-values for each post-event observation ---
        p_vals = _compute_ecdf_pvalues(da_ref, da_post, min_ref)  # (time, y, x)

        # --- Pre-event stability (sigma) ---
        sigma = da_ref.std('time')  # (y, x)
        w_stability = pixel_sigma_weighting(sigma)

        # --- Sign of change (increase vs decrease) ---
        median = da_ref.median('time')
        sign = xr.apply_ufunc(
            np.sign,
            da_post - median,
            dask="parallelized",
            output_dtypes=[np.int8],
        )  # (time, y, x)

        # --- Convert p-values to signed z-scores ---
        z_signed = _pvalues_to_signed_z(p_vals, sign)  # (time, y, x)

        # --- Incidence angle weights ---
        w_incidence = incidence_angle_weight(lia, lia_optimal, lia_width)  # (y, x)

        # --- Polarization weight ---
        w_pol = q_pol.get(pol, 0.5)  # scalar

        # --- Combine weights (same as empirical!) ---
        w_total = combine_weights(
            w_stability,  # (y, x)
            w_incidence,  # (y, x)
            w_pol,        # scalar
        )  # (y, x)

        # --- Weighted combination of z-scores across time ---
        z_combined = _combine_z_across_time(z_signed, w_total)  # (y, x)

        # --- Convert to probability ---
        p_single = _z_to_probability(z_combined, beta=beta, z_pivot=z_pivot)
        probability_list.append(p_single)

    if not probability_list:
        raise ValueError(
            "No valid track/polarization combinations found. "
            "Check data availability and min_ref threshold."
        )

    # --- Combine across tracks/pols (same as empirical!) ---
    log.info(f"Combining {len(probability_list)} probability maps with alpha={combine_alpha}")
    
    probability_combined = log_odds_combine(
        probability_list,
        alpha=combine_alpha,
    )

    return probability_combined


def _combine_z_across_time(z_signed: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """
    Combine z-scores across time dimension using spatial weights.
    
    Parameters
    ----------
    z_signed : xr.DataArray
        Signed z-scores, dims=(time, y, x)
    weights : xr.DataArray
        Spatial weights, dims=(y, x)
    
    Returns
    -------
    xr.DataArray
        Combined z-score, dims=(y, x)
    """
    # Each time observation gets the same spatial weight pattern
    # We're averaging z-scores across time, weighted by pixel reliability
    
    valid = xr.where(np.isfinite(z_signed), 1.0, 0.0)
    
    # Broadcast weights to time dimension
    w_broadcast = weights  # xarray will broadcast automatically
    
    w_eff = w_broadcast * valid
    
    # Weighted mean across time
    num = (w_eff * z_signed).sum("time", skipna=True)
    den = w_eff.sum("time", skipna=True)
    
    z_mean = num / (den + 1e-6)
    
    return z_mean