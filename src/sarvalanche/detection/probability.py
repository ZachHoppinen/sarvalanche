
import numpy as np
import pandas as pd
import xarray as xr

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
    threshold_db: float = 0.75  # dB change where p=0.5
) -> xr.DataArray:
    """
    Convert backscatter change into probability using absolute dB thresholds.

    Parameters
    ----------
    diff : xr.DataArray
        Backscatter change in dB, dims=(y,x) or (pair,y,x)
    logistic_slope : float
        Steepness of the sigmoid. Default 3.0 means:
        - at threshold_db: p = 0.5
        - at threshold_db + 1dB: p ≈ 0.95
    threshold_db : float
        dB change where probability = 0.5 (default 3.0 dB)

    Returns
    -------
    xr.DataArray
        Per-pixel probability of avalanche debris, dims same as `diff`.
    """
    # --- Logistic probability directly on dB values ---
    prob = 1 / (1 + np.exp(-logistic_slope * (diff - threshold_db)))

    # --- Clip to valid probability range ---
    prob = prob.clip(0, 1)

    # --- Wrap as DataArray ---
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

def _z_to_probability(Z_norm: xr.DataArray, beta: float, z_pivot: float) -> xr.DataArray:
    """Convert normalized z-score to probability using sigmoid."""
    P_change = 1 / (1 + np.exp(-beta * (Z_norm - z_pivot)))
    return P_change