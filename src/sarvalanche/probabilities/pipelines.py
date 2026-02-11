
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from sarvalanche.probabilities.features import probability_slope_angle
from sarvalanche.probabilities.features import probability_cell_counts
from sarvalanche.probabilities.features import probability_forest_cover
from sarvalanche.probabilities.features import probability_swe_accumulation

# dense CRF processing
from sarvalanche.probabilities import dense_crf
from sarvalanche.probabilities.dense_crf import run_spatial_crf_densecrf_py38

# filtering functions
from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth


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

def group_classes(ds, cache_dir):
    log.info('Running dense CRF processing')
    # generate U from p_total for dense CRF
    arr = spatial_smooth(ds['p_pixelwise'])
    arr = np.asarray(arr, dtype='<f4')

    P_debris = np.clip(arr, eps, 1 - eps)
    P_background = 1.0 - P_debris

    U = np.stack([
        -np.log(P_background),  # label 0
        -np.log(P_debris)       # label 1
    ], axis=0).astype(np.float32)

    assert U.dtype == np.float32, f'U dtype is {U.dtype} not float32'
    assert U.ndim == 3, f'U has {U.ndim} instead of expected 3 (label, y, x)'
    assert np.isfinite(U).all(), f'U has {(~np.isfinite(U)).sum()} non-finite values'
    assert U.shape[0] == 2, f'U has {U.shape[0]} labels instead of expected 2 of background, debris'

    U_fp = cache_dir.joinpath('arrays', 'U.npy')
    np.save(U_fp, U)
    Q_fp = cache_dir.joinpath('arrays', 'Q.npy')
    log.debug(f'Using output array locations {U_fp} and {Q_fp}')

    dense_crf_script_path = Path(dense_crf.__file__)
    log.debug(f'Dense CRF script found at {dense_crf_script_path}')
    run_spatial_crf_densecrf_py38(U_fp, Q_fp, dense_crf_script_path, iters = 5)

    p_crf = np.load(Q_fp)[1]
    mask = p_crf > 0.5
    mask_da = xr.zeros_like(ds['dem'])
    mask_da.data = mask

    detections, n_labels = filter_pixel_groups(mask_da, min_size=8, return_nlabels=True)
    log.info(f'N labels found in detections: {n_labels}')
    detections.attrs = {'source': 'sarvalance', 'units': 'binary', 'product': 'detection_map'}

    return detections
