from pathlib import Path
import logging

import pandas as pd
import geopandas as gpd

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

# flowpy
from sarvalanche.features.debris_flow_modeling import generate_runcount_alpha_angle

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
        avalanche_date,
        cache_dir,
        resolution = None,
        crs = 'EPSG:4326',
        start_date = None,
        stop_date = None,
        static_fp = None,
        track_gpkg = None,
        overwrite=False,
        job_name=None,
        debug=False):
    """
    Run the SARvalanche detection pipeline for a given AOI and date range.

    Uses SAR backscatter change detection combined with terrain, snow, and
    runout modeling to identify avalanche deposits in Sentinel-1 RTC imagery.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in WGS84 geographic coordinates (lon/lat).
    avalanche_date : str or datetime
        Date of the avalanche event to detect (e.g. '2020-01-11').
    cache_dir : str or Path
        Directory for caching intermediate files and outputs. Created if it
        does not exist.
    resolution : float, optional
        Spatial resolution for processing. Units depend on CRS: meters for
        projected CRS, degrees for geographic. Defaults to 30m for projected
        CRS or 1 arc second (1/3600 degrees) for geographic CRS.
    crs : str or int, optional
        Coordinate reference system for processing (e.g. 'EPSG:32610').
        Defaults to 'EPSG:4326'.
        find_utm_crs(aoi) to get the appropriate zone automatically.
    start_date : str or datetime, optional
        Start of SAR acquisition window. Defaults to 6 Sentinel-1 revisit
        cycles (~72 days) before avalanche_date.
    stop_date : str or datetime, optional
        End of SAR acquisition window. Defaults to 3 Sentinel-1 revisit
        cycles (~36 days) after avalanche_date.
    static_fp : Path, optional
        Path to a pre-built NetCDF containing static layers (DEM, slope,
        aspect, forest cover, LIA, ANF, SWE). If provided, skips downloading
        and computing these layers.
    track_gpkg : Path, optional
        Path to a GeoPackage for caching flowpy debris track outputs. Defaults
        to the same stem as the output NetCDF with a .gpkg extension.
    overwrite : bool, optional
        If True, recompute and overwrite all cached results. Default False.
    job_name : str, optional
        Stem for output filenames. Defaults to avalanche_date as 'YYYY-MM-DD'.
    debug : bool, optional
        If True, enables DEBUG logging for the sarvalanche logger. Default False.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, y, x) containing:
        - detections : binary detection mask after spatial CRF grouping
        - p_pixelwise : combined pixel-wise avalanche probability
        - p_empirical : SAR backscatter change probability
        - p_fcf : forest cover probability
        - p_runout : debris flow runout probability
        - p_slope : slope angle probability
        - p_swe : snow water equivalent accumulation probability
    """

    if debug:
        logging.getLogger('sarvalanche').setLevel(logging.DEBUG)
        log.debug('Debug logging enabled')

    # Initialize timer - this automatically starts tracking total time
    timer = PipelineTimer()

    log.info(f"Arguments: {locals()}")

    # ================================================================
    # Step 1: Validate all input arguments
    # ================================================================
    timer.step('1_validation')

    log.info('Validating arguments')
    avalanche_date = validate_date(avalanche_date)
    aoi = validate_aoi(aoi)

    S1_REVISIT_DAYS = 12
    if start_date is None:
        start_date = avalanche_date - pd.Timedelta(days=6 * S1_REVISIT_DAYS)
        log.info(f'No start date provided. Using {start_date}')
    if stop_date is None:
        stop_date = avalanche_date + pd.Timedelta(days=3 * S1_REVISIT_DAYS)
        log.info(f'No stop date provided. Using {stop_date}')

    start_date, stop_date = validate_start_end(start_date, stop_date)

    crs = validate_crs(crs)

    if resolution is None:
        if crs.is_projected:
            resolution = 30  # meters
        else:
            resolution = 1 / 3600  # 1 arc second in degrees
        log.info(f'No resolution provided. Using: {resolution}')

    resolution = validate_resolution(resolution)

    cache_dir = validate_path(cache_dir, should_exist=None, make_directory=True)
    assert isinstance(overwrite, bool)
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
            log.info('Netcdf found. Overwriting dataset based on overwrite = True.')

        ds = assemble_dataset(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            resolution=resolution,
            crs=crs,
            cache_dir=cache_dir,
            static_layer_nc=static_fp,
            sar_only=False)

        log.info(f'Saving netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc)

    else:
        log.info(f'Found netcdf at {ds_nc}. Loading from cache...')
        ds = load_netcdf_to_dataset(ds_nc)

    # add in flowpy outputs and generate track list
    track_gpkg = ds_nc.with_suffix('.gpkg') if track_gpkg is None else track_gpkg
    missing_flowpy_vars = not all(v in ds.data_vars for v in ['cell_counts', 'runout_angle'])
    needs_flowpy = (
        overwrite
        or not track_gpkg.exists()
        or track_gpkg.stat().st_size == 0
        or missing_flowpy_vars)

    if needs_flowpy:
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg, driver='GPKG')
        if missing_flowpy_vars: export_netcdf(ds, ds_nc, overwrite=True)
    else:
        paths_gdf = gpd.read_file(track_gpkg)

    validate_canonical(ds)

    # 3.5 - Preprocessing
    # rtc pre-processing is a homomorphic total variation based despeckling on each time step for each pol
    print(ds.attrs)
    print(ds.attrs.get('preprocessed'))
    if ds.attrs.get('preprocessed') != 'rtc_tv':
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs['preprocessed'] = 'rtc_tv'

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

    # recheck validitity before exporting results
    validate_canonical(ds)

    log.info(f'Saving final results to {ds_nc}')
    export_netcdf(ds, ds_nc, overwrite=True)

    # Export individual probability layers as GeoTIFFs for visualization
    cache_dir.joinpath("probabilities").mkdir(exist_ok=True)
    for var in ['detections', 'p_pixelwise', 'p_empirical', 'p_fcf', 'p_runout', 'p_slope', 'release_zones', 'distance_mahalanobis']:
        output_tif = cache_dir.joinpath("probabilities", f"{ds_stem}_{var}.tif")
        log.info(f'Exporting {var} to {output_tif.name}')
        ds[var].astype(float).rio.to_raster(output_tif)

    # ================================================================
    # Print timing summary and return results
    # ================================================================
    timer.summary()

    # ── Cleanup: release large in-memory objects before returning ────────────
    # flowpy and SAR stacking leave large arrays allocated; explicit cleanup
    # prevents OOM kills when running multiple zones back-to-back
    import gc
    gc.collect()

    return ds