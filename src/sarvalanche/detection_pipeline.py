from pathlib import Path
import logging

# io functions
from sarvalanche.utils.validation import (
    validate_crs,
    validate_path,
    validate_canonical,
    validate_start_end,
    validate_date,
    validate_resolution,
    validate_aoi)
from sarvalanche.io.dataset import assemble_dataset, load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf

# preprocessing
from sarvalanche.preprocessing.pipelines import preprocess_rtc

# weights and static probabilities
from sarvalanche.weights.pipelines import get_static_weights
from sarvalanche.probabilities.pipelines import get_static_probabilities

# pixelwise terrain, snow, SAR probabilities
from sarvalanche.detection.pixelwise import get_pixelwise_probabilities

# grouping of probabilities
from sarvalanche.probabilities.pipelines import group_classes

# masking
from sarvalanche.masks.pipelines import apply_exclusion_masks

# timing utilities
from sarvalanche.utils.timing import PipelineTimer

# Configure logging at the very top of your main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('asf_search').setLevel(logging.WARNING)  # or logging.ERROR
log = logging.getLogger(__name__)


def run_detection(
        aoi,
        crs,
        resolution,
        start_date,
        stop_date,
        avalanche_date,
        cache_dir,
        overwrite=False,
        job_name=None):
    """
    Run the SARvalanche detection pipeline for a given AOI and date range.

    This pipeline uses backscatter change detection to identify avalanche deposits
    in SAR imagery. The process follows these steps:

    1. Validate inputs
    2. Set up cache directory
    3. Load or assemble SAR dataset
    4. Calculate spatial weights
    5. Calculate static probabilities (terrain, snow)
    6. Calculate pixelwise probabilities using SAR data
    7. Group and refine detections
    8. Export results

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in geographic coordinates (lon/lat).
    crs : str or int
        Target coordinate reference system (e.g., 'EPSG:32610' or 32610).
    resolution : int or float
        Spatial resolution in meters.
    start_date : str | datetime
        Start of SAR acquisition range for analysis.
    stop_date : str | datetime
        End of SAR acquisition range for analysis.
    avalanche_date : str | datetime
        Date of the avalanche event to detect.
    cache_dir : str or Path
        Directory to cache intermediate results and outputs.
    overwrite : bool, optional
        If True, recalculate everything. If False, use cached data. Default is False.
    job_name : str, optional
        Custom name for output files. If None, uses avalanche_date as filename.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, y, x) containing:
        - detections: final detection mask
        - p_pixelwise: pixel-by-pixel probabilities
        - p_empirical, p_fcf, p_runout, p_slope: intermediate probability layers
        - Additional metadata and coordinates
    """

    # Initialize timer - this automatically starts tracking total time
    timer = PipelineTimer()

    log.info(f"Arguments: {locals()}")

    # ================================================================
    # Step 1: Validate all input arguments
    # ================================================================
    timer.step('1_validation')

    log.info('Validating arguments')
    start_date, stop_date = validate_start_end(start_date, stop_date)
    avalanche_date = validate_date(avalanche_date)
    crs = validate_crs(crs)
    resolution = validate_resolution(resolution)
    aoi = validate_aoi(aoi)
    cache_dir = validate_path(cache_dir, should_exist=None, make_directory=True)
    assert overwrite in [True, False]
    log.info(f'Initial validation checks passed')

    # ================================================================
    # Step 2: Set up cache directory structure
    # ================================================================
    timer.step('2_setup_cache')

    for sub_directory_name in ['opera', 'arrays']:
        cache_dir.joinpath(sub_directory_name).mkdir(exist_ok=True)

    ds_stem = f'{avalanche_date.strftime("%Y-%m-%d")}' if job_name is None else f'{job_name}'
    ds_nc = cache_dir.joinpath(ds_stem).with_suffix('.nc')
    log.info(f'Results will be saved to {ds_nc}')

    # ================================================================
    # Step 3: Load or assemble dataset
    # ================================================================
    timer.step('3_load_assemble_dataset')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:
        if not ds_nc.exists() or ds_nc.stat().st_size == 0:
            log.info('Netcdf not found. Assembling dataset now.')
        elif overwrite == True:
            log.info('Netcdf found. Overwriting dataset.')

        ds = assemble_dataset(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            resolution=resolution,
            crs=crs,
            cache_dir=cache_dir)

        log.info(f'Saving netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc)
    else:
        log.info(f'Found netcdf at {ds_nc}. Loading from cache...')
        ds = load_netcdf_to_dataset(ds_nc)

    validate_canonical(ds)

    # 3.5 - Preprocessing
    # rtc pre-processing is a homomorphic total variation based despeckling on each time step for each pol
    ds = preprocess_rtc(ds, tv_weight = 0.5)

    # ================================================================
    # Step 4: Calculate spatial domain weights
    # ================================================================
    timer.step('4_calculate_weights')

    ds = get_static_weights(ds, avalanche_date)

    # ================================================================
    # Step 5: Calculate static probabilities (terrain, snow, etc.)
    # ================================================================
    timer.step('5_static_probabilities')

    # Generate non-SAR probabilities based on terrain and snow conditions
    # TODO: Add water, urban, cropland masking in future versions
    ds = get_static_probabilities(ds, avalanche_date)

    # ================================================================
    # Step 6: Calculate pixel-by-pixel probabilities using SAR data
    # ================================================================
    timer.step('6_pixelwise_probabilities')

    # Combine SAR backscatter changes with static probabilities
    ds['p_pixelwise'] = get_pixelwise_probabilities(ds, avalanche_date)

    # Step 6.5: Apply exclusion masks
    timer.step('6.5_apply_masks')
    ds = apply_exclusion_masks(ds)


    # ================================================================
    # Step 7: Group and refine detections
    # ================================================================
    timer.step('7_group_detections')

    # Increase likelihood of neighboring pixels (spatial smoothing)
    # rename to dense_crf...
    ds['detections'] = group_classes(ds['p_pixelwise'], cache_dir)

    # ================================================================
    # Step 8: Validate and export final results
    # ================================================================
    timer.step('8_export_results')

    validate_canonical(ds)

    log.info(f'Saving final results to {ds_nc}')
    export_netcdf(ds, ds_nc, overwrite=True)

    # Export individual probability layers as GeoTIFFs for visualization
    for var in ['detections', 'p_pixelwise', 'p_empirical', 'p_fcf', 'p_runout', 'p_slope']:
        output_tif = cache_dir.joinpath(f"{ds_stem}_{var}.tif")
        log.info(f'Exporting {var} to {output_tif.name}')
        ds[var].astype(float).rio.to_raster(output_tif)

    # ================================================================
    # Print timing summary and return results
    # ================================================================
    timer.summary()

    return ds