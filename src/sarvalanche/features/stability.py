
import xarray as xr

def pixel_sigma_weighting(
        sigma: xr.DataArray,
        sigma_threshold: float = 2.0,
) -> xr.DataArray:
    """
    Weight pixels by temporal stability with saturation.

    Parameters
    ----------
    sigma : xr.DataArray
        Temporal std of backscatter in dB
    sigma_threshold : float, default=2.0
        Sigma value (dB) where weight = 0.5
        Below this: high weight, above this: low weight
    """
    # Logistic decay: smooth transition, bounded [0, 1]
    weights = 1.0 / (1.0 + (sigma / sigma_threshold) ** 2)

    return weights.rename('w_sigma')