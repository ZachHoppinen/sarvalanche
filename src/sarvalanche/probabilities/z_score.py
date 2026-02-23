import numpy as np
import xarray as xr
from scipy.stats import norm

def z_score_to_probability(distance: xr.DataArray, threshold: float = 2.0) -> xr.DataArray:
    """
    Convert z-scores to probability of anomalously HIGH backscatter,
    rescaled so that z=threshold maps to p=0.5.

    Uses a logistic transform on the z-score centered at threshold,
    so the output is intuitive: p>0.5 means "above typical detection threshold".

    threshold=2.0 (default) means z=+2 → p=0.5, z=+3 → p~0.88, z=+4 → p~0.98
    """
    prob = xr.apply_ufunc(
        lambda z: 1 / (1 + np.exp(-(z - threshold))),
        distance,
    )
    prob.name = 'ml_probability'
    prob.attrs = {**distance.attrs, 'units': 'probability [0, 1]', 'threshold': threshold}
    return prob