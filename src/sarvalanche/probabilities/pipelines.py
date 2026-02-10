import logging

from sarvalanche.probabilities.features import probability_slope_angle
from sarvalanche.probabilities.features import probability_cell_counts
from sarvalanche.probabilities.features import probability_forest_cover
from sarvalanche.probabilities.features import probability_swe_accumulation

from sarvalanche.utils.constants import eps

log = logging.getLogger(__name__)

def get_static_probabilities(ds, avalanche_date):
    # --- 6. Compute forest cover probability ---
    log.info('Calculating forest cover probability')
    ds['p_fcf'] = probability_forest_cover(ds['fcf'])

    # --- 7. Compute avalanche model cell counts probability ---
    log.info('Calculating runout cell based probability')
    ds['p_runout'] = probability_cell_counts(ds['cell_counts'])

    # --- 8. Compute slope angle probability of debris ---
    log.info('Calculating slope-angle based probability')
    ds['p_slope'] = probability_slope_angle(ds['slope'])

    # --- 9. Compute swe accumulation probability of debris ---
    log.info('Calculating swe accumulation based probability')
    ds['p_swe'] = probability_swe_accumulation(ds['swe'], avalanche_date)

    for d in ['p_fcf', 'p_runout', 'p_slope', 'p_swe']:
        ds[d].attrs = {'source': 'sarvalance', 'units': 'percentage', 'product': 'pixel_wise_probability'}

    return ds