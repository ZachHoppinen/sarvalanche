
import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import expit

def probability_backscatter_change(
    diff: xr.DataArray,
    logistic_slope: float = 5.0,
    threshold_db: float = 0.75
) -> xr.DataArray:
    """Convert backscatter change to probability using stable sigmoid."""

    # expit(x) = 1 / (1 + exp(-x)) but numerically stable
    prob = xr.apply_ufunc(
        expit,
        (diff - threshold_db)* logistic_slope,
        dask="parallelized",
        output_dtypes=[float],
    )

    prob = prob.clip(0, 1)

    return xr.DataArray(prob, dims=diff.dims, coords=diff.coords, name="p_avalanche")

def probability_forest_cover(fcf: xr.DataArray,
                             midpoint: float = 50.0,
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
    avalanche_date: pd.Timestamp,
    accumulation_days: int = 7,
    midpoint: float = 0.0,
    slope: float = 100.0
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