
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# io functions
from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs, validate_path, validate_canonical
from sarvalanche.io.dataset import assemble_dataset, load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf

# backscatter based probabilities
from sarvalanche.detection.probability import calculate_emperical_backscatter_probability
from sarvalanche.detection.ecdf import calculate_ecdf_backscatter_probability

# probability functions
from sarvalanche.detection.probability import probability_slope_angle
from sarvalanche.detection.probability import probability_cell_counts
from sarvalanche.detection.probability import probability_forest_cover
from sarvalanche.detection.probability import probability_swe_accumulation
from sarvalanche.detection.probability import weighted_geometric_mean

# dense CRF processing
from sarvalanche.detection import dense_crf
from sarvalanche.detection.dense_crf import run_spatial_crf_densecrf_py38

# filtering functions
from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth

from sarvalanche.utils.constants import eps

def detect_avalanche_debris(
        aoi,
        crs,
        resolution,
        start_date,
        stop_date,
        avalanche_date,
        cache_dir,
        overwrite = False):

    crs = validate_crs(crs)
    # this should be generalized to any CRS and renamed
    resolution_deg = resolution_to_degrees(resolution, crs)

    cache_dir = validate_path(cache_dir, make_directory = True)
    cache_dir.joinpath('opera').mkdir(exist_ok = True)
    cache_dir.joinpath('arrays').mkdir(exist_ok = True)

    ds_nc = cache_dir.joinpath(f'{avalanche_date}.nc')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:

        ds = assemble_dataset(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            resolution=resolution_deg,
            crs=crs,
            cache_dir=cache_dir
        )
        export_netcdf(ds, ds_nc)
    else:
        ds = load_netcdf_to_dataset(ds_nc)

    validate_canonical(ds)

    # one method based on weighted backscatter changes
    ds['p_emperical'] = calculate_emperical_backscatter_probability(ds, avalanche_date, smooth_method=None)

    # method based on probability of change from pre-event distribution
    ds['p_ecdf'] = calculate_ecdf_backscatter_probability(ds, avalanche_date)

    # --- 6. Compute forest cover probability ---
    ds['p_fcf'] = probability_forest_cover(ds['fcf'], midpoint=50, slope = 0.1)

    # --- 7. Compute avalanche model cell counts probability ---
    ds['p_runout'] = probability_cell_counts(ds['cell_counts'])

    # --- 8. Compute slope angle probability of debris ---
    ds['p_slope'] = probability_slope_angle(ds['slope'])

    # --- 9. Compute swe accumulation probability of debris ---
    ds['p_swe'] = probability_swe_accumulation(ds['swe'], avalanche_date, midpoint = 0.0, slope = 100.0)

    factors = [ds['p_emperical'], ds['p_ecdf'], ds['p_fcf'], ds['p_runout'], ds['p_slope'], ds['p_swe']]
    weights = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

    p_total = weighted_geometric_mean(factors, weights)
    p_total = p_total.where(~p_total.isnull(), 0)

    ds['p_pixelwise'] = p_total
    for d in ['p_slope', 'p_runout', 'p_fcf', 'p_ecdf', 'p_emperical', 'p_pixelwise', 'p_swe']:
        ds[d].attrs = {'source': 'sarvalance', 'units': 'percentage', 'product': 'pixel_wise_probability'}

    # generate U from p_total for dense CRF
    arr = spatial_smooth(p_total)
    arr = np.asarray(arr, dtype='<f4')

    P_debris = np.clip(arr, eps, 1 - eps)
    P_background = 1.0 - P_debris

    U = np.stack([
        -np.log(P_background),  # label 0
        -np.log(P_debris)       # label 1
    ], axis=0).astype(np.float32)

    assert U.dtype == np.float32
    assert U.ndim == 3
    assert np.isfinite(U).all()
    assert U.shape[0] == 2

    U_fp = cache_dir.joinpath('arrays', 'U.npy')
    np.save(U_fp, U)
    Q_fp = cache_dir.joinpath('arrays', 'Q.npy')

    dense_crf_script_path = Path(dense_crf.__file__)
    run_spatial_crf_densecrf_py38(U_fp, Q_fp, dense_crf_script_path, iters = 5)

    p_crf = np.load(Q_fp)[1]
    mask = p_crf > 0.5
    mask_da = xr.zeros_like(ds['dem'])
    mask_da.data = mask

    ds['detections'] = filter_pixel_groups(mask_da, min_size=8)
    ds['detections'].attrs = {'source': 'sarvalance', 'units': 'binary', 'product': 'detection_map'}

    validate_canonical(ds)

    export_netcdf(ds, ds_nc, overwrite = True)

    return ds
