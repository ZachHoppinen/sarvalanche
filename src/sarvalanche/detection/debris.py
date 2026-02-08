
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
from sarvalanche.detection.backscatter_detections import calculate_empirical_backscatter_probability
# from sarvalanche.detection.backscatter_detections import calculate_ecdf_backscatter_probability

# weights
from sarvalanche.weights.combinations import get_static_weights
from sarvalanche.probabilities.combine import combine_probabilities, get_static_probabilities

# dense CRF processing
from sarvalanche.detection import dense_crf
from sarvalanche.detection.dense_crf import run_spatial_crf_densecrf_py38

# filtering functions
from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth

from sarvalanche.utils.constants import eps

import logging

# Configure logging at the very top of your main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('asf_search').setLevel(logging.WARNING)  # or logging.ERROR
log = logging.getLogger(__name__)

def detect_avalanche_debris(
        aoi,
        crs,
        resolution,
        start_date,
        stop_date,
        avalanche_date,
        cache_dir,
        overwrite = False):

    log.info(f"Arguments: {locals()}")

    start_date, stop_date = validate_start_end(start_date, stop_date)
    avalanche_date = validate_date(avalanche_date)


    crs = validate_crs(crs)
    # this should be generalized to any CRS and renamed
    resolution_deg = resolution_to_degrees(resolution, crs)

    cache_dir = validate_path(cache_dir, make_directory = True)
    cache_dir.joinpath('opera').mkdir(exist_ok = True)
    cache_dir.joinpath('arrays').mkdir(exist_ok = True)

    ds_nc = cache_dir.joinpath(f'{avalanche_date}.nc')

    log.info(f'Initial validation checks passed')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:
        log.info('Netcdf not found. Assembling dataset now.')
        ds = assemble_dataset(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            resolution=resolution_deg,
            crs=crs,
            cache_dir=cache_dir
        )
        log.info(f'Saving netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc)
    else:
        log.info(f'Found netcdf at {ds_nc}. Loading...')
        ds = load_netcdf_to_dataset(ds_nc)

    validate_canonical(ds)

    # next generate spatial domain weights
    ds = get_static_weights(ds, avalanche_date)

    # one method based on weighted backscatter changes
    log.info('Calculating emperical backscatter change probability')
    ds['p_emperical'] = calculate_empirical_backscatter_probability(ds, avalanche_date, smooth_method=None)

    # method based on probability of change from pre-event distribution
    # log.info('Calculating distribution based probability')
    # ds['p_ecdf'] = calculate_ecdf_backscatter_probability(ds, avalanche_date)

    ds = get_static_probabilities(ds, avalanche_date)

    factors = [ds['p_emperical'], ds['p_fcf'], ds['p_runout'], ds['p_slope'], ds['p_swe']]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    log.info('Running weighted geometric mean')
    log.debug(f'Weighing with {weights} for backscatter change, distribution, fcf, runout, slope, swe change')

    factors_stacked = xr.concat(factors, dim='factor')  # Stack the list
    weights_da = xr.DataArray(weights, dims='factor')
    p_total = combine_probabilities(
        factors_stacked,
        dim='factor',
        method='product',  # Geometric mean
        weights=weights_da
    )

    p_total = p_total.where(~p_total.isnull(), 0)

    ds['p_pixelwise'] = p_total
    ds['p_pixelwise'].attrs = {'source': 'sarvalance', 'units': 'percentage', 'product': 'pixel_wise_probability'}

    log.info('Running dense CRF processing')
    # generate U from p_total for dense CRF
    arr = spatial_smooth(p_total)
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

    ds['detections'], n_labels = filter_pixel_groups(mask_da, min_size=8, return_nlabels=True)
    log.info(f'N labels found in detections: {n_labels}')
    ds['detections'].attrs = {'source': 'sarvalance', 'units': 'binary', 'product': 'detection_map'}

    validate_canonical(ds)

    log.info(f'Saving final results to {ds_nc}')
    export_netcdf(ds, ds_nc, overwrite = True)

    return ds
