
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.features.backscatter_change import (
    backscatter_changes_crossing_date,
    backscatter_change_weighted_mean,
)

def calculate_emperical_backscatter_probability(
    ds: xr.Dataset,
    avalanche_date,
    *,
    polarizations=("VV", "VH"),
    smooth_method=None,
    tau_days=24,
    tau_variability=20.0,
    incidence_power=0.0,
    combine_alpha=0.5,
):
    """
    Compute avalanche debris probability from SAR backscatter changes.

    For each (track, polarization) pair:
      1. Convert backscatter to dB
      2. Spatially smooth
      3. Compute pre/post-avalanche backscatter changes
      4. Aggregate changes using temporal, stability, and incidence weighting
      5. Convert to probability

    All per-track/pol probabilities are then fused using log-odds
    combination with optional shrinkage toward 0.5.

    Parameters
    ----------
    ds : xr.Dataset
        Canonical SAR dataset containing:
        - VV, VH backscatter (time, y, x)
        - track (time)
        - lia (static_track, y, x)
    avalanche_date : str or datetime-like
        Date separating pre/post-event acquisitions.
    polarizations : tuple[str]
        Polarizations to include (default: ("VV", "VH")).
    smooth_method : str
        Spatial smoothing method passed to `spatial_smooth`.
        (None = no smoothing)
    tau_days : float
        Temporal decay scale (days) for weighting.
    tau_variability : float
        Variability decay scale for stability weighting.
    incidence_power : float
        Power applied to incidence angle weighting.
    combine_alpha : float
        Shrinkage factor toward 0.5 when combining probabilities
        (0 = ignore, 1 = full strength).

    Returns
    -------
    xr.DataArray
        Combined backscatter-change probability in [0, 1].
    """

    tracks = np.unique(ds.track.values)
    p_delta_list: list[xr.DataArray] = []

    for track in tracks:
        lia = ds["lia"].sel(static_track=track)

        for pol in polarizations:
            if pol not in ds:
                continue

            # --- 1. Select, convert to dB, smooth ---
            da = ds[pol].sel(time=ds.track == track)
            da_db = linear_to_dB(da)
            if smooth_method is not None:
                da_db = spatial_smooth(da_db, method=smooth_method)

            # --- 2. Backscatter change across avalanche date ---
            diffs = backscatter_changes_crossing_date(
                da_db, avalanche_date
            )

            # --- 3. Weighted aggregation ---
            weighted_mean = backscatter_change_weighted_mean(
                diffs,
                da_db,
                tau_days=tau_days,
                tau_variability=tau_variability,
                local_incidence_angle=lia,
                incidence_power=incidence_power,
            )

            # --- 4. Probability mapping ---
            p_delta = probability_backscatter_change(weighted_mean)
            p_delta_list.append(p_delta)

    if not p_delta_list:
        raise ValueError("No backscatter probabilities were computed")

    # --- 5. Combine across tracks / pols ---
    p_delta_combined = log_odds_combine(
        p_delta_list,
        alpha=combine_alpha,
    )

    return p_delta_combined

def weighted_geometric_mean(
    probs: list[xr.DataArray],
    weights: list[float] = None,
    eps: float = 1e-6,
    normalize: bool = False
) -> xr.DataArray:
    """
    Combine multiple probability DataArrays using a weighted geometric mean.

    Parameters
    ----------
    probs : list of xr.DataArray
        Probabilities in [0, 1] to combine. All must have the same shape/dims.
    weights : list of float, optional
        Weight for each probability array. Defaults to equal weighting.
    eps : float
        Small value to avoid log(0).
    normalize : bool
        If True, scales final probabilities to [0, 1].

    Returns
    -------
    xr.DataArray
        Combined probability array.
    """
    if weights is None:
        weights = [1.0] * len(probs)
    if len(weights) != len(probs):
        raise ValueError("Length of weights must match number of probability arrays.")

    # Convert probabilities to log space with numerical stability
    log_probs = [w * np.log(np.clip(p, eps, 1.0)) for p, w in zip(probs, weights)]

    # Sum weighted logs
    log_total = sum(log_probs)

    # Back to probability space
    combined = np.exp(log_total)

    # Optional normalization to [0,1]
    if normalize:
        combined = combined / combined.max()

    # Preserve coords/dims from first array
    out = xr.DataArray(
        combined,
        dims=probs[0].dims,
        coords=probs[0].coords,
        name="p_total_geomean"
    )

    return out

def log_odds_combine(
    probs,
    dim=None,
    weights=None,
    alpha=1.0,
    eps=1e-6,
):
    """
    Combine probabilities by summing log-odds with optional shrinkage and weighting.

    Parameters
    ----------
    probs : list[xr.DataArray] or xr.DataArray
        Probabilities in [0, 1].
        If a list, concatenated along new 'stack' dimension.
    dim : str, optional
        Dimension to combine over (required if probs is a DataArray).
    weights : list[float], optional
        Per-probability weights (applied in log-odds space). Must match number of probs.
    alpha : float
        Shrinkage factor toward 0.5 (0 = ignore, 1 = full strength).
    eps : float
        Numerical stability constant.

    Returns
    -------
    xr.DataArray
        Combined probability in [0, 1].
    """

    # --- concatenate list of DataArrays if needed ---
    if isinstance(probs, list):
        probs = xr.concat(probs, dim="stack")
        dim = "stack"

    # --- shrink toward 0.5 (confidence control) ---
    if alpha is not None:
        probs = 0.5 + alpha * (probs - 0.5)

    # --- log-odds transform with optional weighting ---
    if weights is None:
        log_odds = np.log((probs + eps) / (1.0 - probs + eps))
    else:
        if len(weights) != probs.sizes[dim]:
            raise ValueError(f"Length of weights ({len(weights)}) must match size of dim '{dim}' ({probs.sizes[dim]})")
        # multiply each slice along dim by its weight
        slices = [probs.isel({dim: i}) for i in range(probs.sizes[dim])]
        log_odds_slices = [w * np.log((p + eps) / (1 - p + eps)) for p, w in zip(slices, weights)]
        # stack back
        log_odds = xr.concat(log_odds_slices, dim=dim)

    # --- combine along dim ---
    log_odds_sum = log_odds.sum(dim, skipna=True)

    # --- back to probability ---
    return 1.0 / (1.0 + np.exp(-log_odds_sum))

def probability_backscatter_change(
    diff: xr.DataArray,
    logistic_slope: float = 3.0,
    logistic_midpoint: float = 0.5
) -> xr.DataArray:
    """
    Convert weighted backscatter change into a per-pixel probability of avalanche debris
    using a logistic (sigmoid) function.

    Parameters
    ----------
    diff : xr.DataArray
        Weighted backscatter change (Δσ⁰), dims=(y,x) or (pair,y,x)
    logistic_slope : float
        Steepness of the sigmoid function. Larger values → sharper transition.
    logistic_midpoint : float
        Normalized Δσ⁰ value (0-1) where probability = 0.5.

    Returns
    -------
    xr.DataArray
        Per-pixel probability of avalanche debris, dims same as `diff`.
    """

    # --- 1. Normalize delta to 0-1 ---
    delta_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)

    # --- 2. Logistic probability ---
    prob = 1 / (1 + np.exp(-logistic_slope * (delta_norm - logistic_midpoint)))

    # --- 3. Clip to valid probability range ---
    prob = prob.clip(0, 1)

    # --- 4. Wrap as DataArray ---
    prob_da = xr.DataArray(prob, dims=diff.dims, coords=diff.coords, name="p_avalanche")

    return prob_da

def probability_forest_cover(fcf: xr.DataArray,
                             midpoint: float = 30.0,
                             slope: float = 0.1) -> xr.DataArray:
    """
    Convert forest cover fraction (0-100) into probability of avalanche debris.

    Parameters
    ----------
    fcf : xr.DataArray
        Forest cover fraction (0-100), dims=(y,x)
    midpoint : float
        Forest cover fraction at which probability = 0.5
    slope : float
        Controls how fast probability drops with increasing forest

    Returns
    -------
    xr.DataArray
        Probability of avalanche debris based on forest, dims=(y,x)
    """

    # Convert 0-100 FCF to probability: low forest = high probability
    prob = 1 / (1 + np.exp(slope * (fcf - midpoint)))

    # Clip to [0,1] just in case
    prob = prob.clip(0, 1)

    prob_da = xr.DataArray(prob, dims=fcf.dims, coords=fcf.coords, name="p_avalanche_forest")

    return prob_da

def probability_cell_counts(cell_counts: xr.DataArray,
                            midpoint: float = 200,
                            slope: float = 7,
                            use_log: bool = True) -> xr.DataArray:
    """
    Convert avalanche model cell counts into probability of avalanche debris.

    Parameters
    ----------
    cell_counts : xr.DataArray
        Number of modelled avalanche pixels that could hit each pixel
    midpoint : float
        Cell count (or log count if use_log=True) at which probability = 0.5
    slope : float
        Steepness of sigmoid: higher → sharper transition
    use_log : bool
        Whether to apply log scaling (log1p) to cell_counts before mapping

    Returns
    -------
    xr.DataArray
        Probability map (0-1) for debris likelihood based on model
    """

    if use_log:
        counts_scaled = np.log1p(cell_counts)  # log(1 + x)
        midpoint_scaled = np.log1p(midpoint)
    else:
        counts_scaled = cell_counts
        midpoint_scaled = midpoint

    # Logistic mapping
    prob = 1 / (1 + np.exp(-slope * (counts_scaled - midpoint_scaled)))

    # Clip to [0,1]
    prob = prob.clip(0, 1)

    prob_da = xr.DataArray(prob,
                           dims=cell_counts.dims,
                           coords=cell_counts.coords,
                           name="p_avalanche_cells")
    return prob_da

def probability_slope_angle(slope_angle: xr.DataArray,
                             midpoint: float = 25.0,
                             slope: float = 1.0) -> xr.DataArray:
    """
    Convert slope in degrees (0-90) into probability of avalanche debris.

    Parameters
    ----------
    slope : xr.DataArray
        Slope in degrees (0-100), dims=(y,x)
    midpoint : float
        Forest cover fraction at which probability = 0.5
    slope : float
        Controls how fast probability drops with increasing forest

    Returns
    -------
    xr.DataArray
        Probability of avalanche debris based on forest, dims=(y,x)
    """
    if slope_angle.max() < 4 * np.pi: slope_angle = np.rad2deg(slope_angle)

    # Convert 0-90 slope to probability: low slope = high probability
    prob = 1 / (1 + np.exp(slope * (slope_angle - midpoint)))

    # Clip to [0,1] just in case
    prob = prob.clip(0, 1)

    prob_da = xr.DataArray(prob, dims=slope_angle.dims, coords=slope_angle.coords, name="p_avalanche_slope")

    return prob_da

def probability_swe_accumulation(
    swe: xr.Dataset,
    avalanche_date: str,
    accumulation_days: int = 7,
    midpoint: float = 100.0,
    slope: float = 0.05
) -> xr.DataArray:
    """
    Calculate probability of avalanche based on SWE accumulation over recent days.

    Parameters
    ----------
    swe : xr.Dataarray
        Dataarray of SWE model results, dims=(time, y, x)
    avalanche_date : str
        Date of avalanche (or date to evaluate), format 'YYYY-MM-DD'
    accumulation_days : int
        Number of days to look back (5-7 typical), default=7
    midpoint : float
        SWE accumulation (mm) at which probability = 0.5
        Typical values: 50-150 mm depending on region
    slope : float
        Controls how fast probability increases with accumulation
        Higher = steeper transition

    Returns
    -------
    xr.DataArray
        Probability of avalanche based on SWE accumulation, dims=(y, x)
    """

    # Convert avalanche_date to datetime
    aval_date = pd.to_datetime(avalanche_date)

    # Calculate start date for accumulation period
    start_date = aval_date - pd.Timedelta(days=accumulation_days)

    # Get SWE at avalanche date and at start of accumulation period
    swe_end = swe.sel(snowmodel_time=aval_date, method='nearest')
    swe_start = swe.sel(snowmodel_time=start_date, method='nearest')

    # Calculate accumulation (change in SWE)
    swe_accumulation = swe_end - swe_start

    # Handle negative accumulation (melt) - set to 0
    swe_accumulation = swe_accumulation.clip(min=0)

    # Convert accumulation to probability using logistic function
    # High accumulation = high probability
    prob = 1 / (1 + np.exp(-slope * (swe_accumulation - midpoint)))

    # Clip to [0,1] just in case
    prob = prob.clip(0, 1)

    # Create DataArray with metadata
    prob_da = xr.DataArray(
        prob,
        dims=swe_accumulation.dims,
        coords=swe_accumulation.coords,
        name="p_avalanche_swe",
        attrs={
            'long_name': 'Avalanche probability based on SWE accumulation',
            'units': 'probability',
            'avalanche_date': avalanche_date,
            'accumulation_days': accumulation_days,
            'midpoint_mm': midpoint,
            'slope': slope
        }
    )

    return prob_da