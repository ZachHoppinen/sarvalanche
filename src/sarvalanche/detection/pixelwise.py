from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# io functions
from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs, validate_path, validate_canonical, validate_start_end, validate_date
from sarvalanche.io.dataset import assemble_dataset, load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf

# backscatter based probabilities
from sarvalanche.detection.backscatter_change import calculate_empirical_backscatter_probability
# from sarvalanche.detection.backscatter_detections import calculate_ecdf_backscatter_probability

# ml mahalanobis
from sarvalanche.detection.mahalanobis import calculate_ml_distances


# probability combinations
from sarvalanche.probabilities.combine import combine_probabilities


import logging

log = logging.getLogger(__name__)

def get_pixelwise_probabilities(
        ds,
        avalanche_date):

    # one method based on weighted backscatter changes
    log.info('Calculating empirical backscatter change probability')
    ds['p_empirical'] = calculate_empirical_backscatter_probability(ds,
                                                                    avalanche_date,
                                                                    smooth_method=None,
                                                                    use_agreement_boosting=True,
                                                                    agreement_strength=0.8,
                                                                    min_prob_threshold=0.2)

    # another based on ml predicted vs observed backscatter (mahalanobis distance)
    ds['distance_mahalanobis'] = calculate_ml_distances(ds, avalanche_date)

    # Static factors are the "prior" - how likely is avalanche here in general?
    p_prior = combine_probabilities(
        xr.concat([ds['p_fcf'], ds['p_runout'], ds['p_slope'], ds['p_swe']], dim='factor'),
        dim='factor',
        method='weighted_mean'
    )

    # Empirical is the "likelihood" - did we detect something?
    p_likelihood = xr.ufuncs.maximum(ds['p_empirical'], ds['distance_mahalanobis'])

    # Standard Bayesian update (allows increases and decreases)
    p_bayesian = p_prior * p_likelihood / (
        p_prior * p_likelihood + (1 - p_prior) * (1 - p_likelihood)
    )

    # Take minimum: this caps the result at the prior probability
    # If Bayesian update > prior: use prior (don't increase)
    # If Bayesian update < prior: use Bayesian (allow decrease)
    p_pixelwise = xr.ufuncs.minimum(p_likelihood, p_bayesian)

    # Hard constraint: if p_runout is 0, force p_pixelwise to 0
    p_pixelwise = p_pixelwise.where(ds['p_runout'] > 0.01, 0)

    log.debug(f'Filling NaN values in p_pixelwise with 0')
    p_pixelwise = p_pixelwise.where(~p_pixelwise.isnull(), 0)

    p_pixelwise.attrs = {'source': 'sarvalanche', 'units': 'percentage', 'product': 'pixel_wise_probability'}

    is_clipped = p_bayesian < p_likelihood
    frac = is_clipped.mean().item()
    log.debug(f'p_bayesian < p_likelihood in {frac*100:.1f}% of pixels')

    # Are clipped pixels concentrated in high-prior areas?
    if frac > 0:
        mean_prior_clipped = p_prior.where(is_clipped).mean().item()
        mean_prior_unclipped = p_prior.where(~is_clipped).mean().item()
        log.debug(f'Mean p_prior where clipped: {mean_prior_clipped:.3f}, unclipped: {mean_prior_unclipped:.3f}')

    return p_pixelwise
