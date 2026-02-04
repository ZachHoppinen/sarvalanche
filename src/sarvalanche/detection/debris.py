
from pathlib import Path
import numpy as np
import xarray as xr

# io functions
from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs, validate_path, validate_canonical
from sarvalanche.io.dataset import assemble_dataset

# flowpy model run
from sarvalanche.masks.debris_flow_modeling import generate_release_mask, run_flowpy_on_mask, attach_flowpy_outputs
from sarvalanche.utils.projections import area_m2_to_pixels

# prepare backscatter
from sarvalanche.features.backscatter_change import backscatter_changes_crossing_date
from sarvalanche.features.backscatter_change import backscatter_change_weighted_mean

from sarvalanche.detection.probability import calculate_backscatter_probability

# probability functions
from sarvalanche.detection.probability import probability_backscatter_change
from sarvalanche.detection.probability import probability_slope_angle
from sarvalanche.detection.probability import probability_cell_counts
from sarvalanche.detection.probability import probability_forest_cover
from sarvalanche.detection.probability import weighted_geometric_mean, log_odds_combine

# dense CRF processing
from sarvalanche.detection import dense_crf
from sarvalanche.detection.dense_crf import run_spatial_crf_densecrf_py38

# filtering functions
from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth

from sarvalanche.utils.constants import eps

def detect_avalanche_debris(aoi, crs, resolution, start_date, stop_date, avalanche_date, cache_dir):
    #TODO pull all IO/mask generation to different function and just give dataset.

    crs = validate_crs(crs)
    # this should be generalized to any CRS and renamed
    resolution_deg = resolution_to_degrees(resolution, crs)

    cache_dir = validate_path(cache_dir, make_directory = True)
    cache_dir.joinpath('opera').mkdir(exist_ok = True)
    cache_dir.joinpath('arrays').mkdir(exist_ok = True)

    ds = assemble_dataset(
        aoi=aoi,
        start_date=start_date,
        stop_date=stop_date,
        resolution=resolution_deg,
        crs=crs,
        cache_dir=cache_dir
    )

    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())

    min_release_area_m2 = 300 *300 # meteers
    min_release_pixels = area_m2_to_pixels(dem_proj, min_release_area_m2)

    # generate start area
    release_mask = generate_release_mask(
        slope=ds['slope'],
        forest_cover=ds['fcf'],
        min_slope_deg=35,
        max_slope_deg=45,
        max_fcf=10,
        min_group_size=min_release_pixels,
        smooth=True,
        reference=dem_proj
    )

    # run FlowPy
    cell_counts_da, runout_angle_da = run_flowpy_on_mask(
        dem=dem_proj,
        release_mask=release_mask,
        alpha=25,
        reference=dem_proj
    )

    # Step 3: Attach to dataset
    ds = attach_flowpy_outputs(ds, cell_counts_da, runout_angle_da)

    validate_canonical(ds)

    p_delta_combined = calculate_backscatter_probability(ds, avalanche_date)

    # --- 6. Compute forest cover probability ---
    p_forest = probability_forest_cover(ds['fcf'])

    # --- 7. Compute avalanche model cell counts probability ---
    p_cells = probability_cell_counts(ds['cell_counts'])

    # --- 8. Compute slope angle probability of debris ---
    p_slope = probability_slope_angle(ds['slope'])

    factors = [p_delta_combined, p_forest, p_cells, p_slope]
    weights = [1.0, 1.0, 1.0, 1.0]

    p_total = weighted_geometric_mean(factors, weights)
    p_total = p_total.where(~p_total.isnull(), 0)

    ds['p_total_pixelwise'] = p_total
    ds['p_total_pixelwise'].attrs = {'source': 'sarvalance', 'units': 'percentage', 'product': 'pixel_wise_probability'}

    # generate U from p_total for dense CRF
    arr = np.asarray(p_total, dtype='<f4')

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

    # validate_canonical(ds)

    return ds
