
import numpy as np
import xarray as xr

def probability_backscatter_change(
    diff: xr.DataArray,
    logistic_slope: float = 5.0,
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
                            slope: float = 0.01,
                            use_log: bool = False) -> xr.DataArray:
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
                             midpoint: float = 30.0,
                             slope: float = 0.5) -> xr.DataArray:
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