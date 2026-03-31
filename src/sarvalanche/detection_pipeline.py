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

# timing utilities
from sarvalanche.utils.timing import PipelineTimer

# Configure logging at the very top of your main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('asf_search').setLevel(logging.WARNING)  # or logging.ERROR
logging.getLogger('rasterio.session').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def prepare_dataset(
        aoi,
        cache_dir,
        avalanche_date=None,
        resolution=None,
        crs='EPSG:4326',
        start_date=None,
        stop_date=None,
        static_fp=None,
        track_gpkg=None,
        overwrite=False,
        job_name=None,
        debug=False):
    """
    Run steps 1–3.5 of the pipeline: validate inputs, assemble the SAR +
    static dataset, compute FlowPy runout, and preprocess (TV despeckle).

    Returns the preprocessed xr.Dataset ready for detection or ML training.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in WGS84 geographic coordinates (lon/lat).
    cache_dir : str or Path
        Directory for caching intermediate files and outputs.
    avalanche_date : str or datetime, optional
        Date of the avalanche event. Used for default start/stop window and
        job naming. If None, start_date and stop_date must be provided.
    resolution : float, optional
        Spatial resolution. Defaults to 30m (projected) or 1 arcsec (geographic).
    crs : str or int, optional
        Target CRS. Default 'EPSG:4326'.
    start_date, stop_date : str or datetime, optional
        SAR acquisition window. Default: 6 revisit cycles before / 3 after
        avalanche_date.
    static_fp : Path, optional
        Pre-built static layers NetCDF.
    track_gpkg : Path, optional
        GeoPackage for caching FlowPy debris track outputs.
    overwrite : bool
        Recompute and overwrite cached results. Default False.
    job_name : str, optional
        Stem for output filenames.
    debug : bool
        Enable DEBUG logging. Default False.

    Returns
    -------
    ds : xr.Dataset
        Preprocessed dataset with SAR (TV-despeckled, dB), static layers,
        and FlowPy outputs.
    track_gpkg_path : Path
        Path to the FlowPy avalanche paths GeoPackage. Load with
        ``gpd.read_file(track_gpkg_path)`` to get the path geometries.
    """
    if debug:
        logging.getLogger('sarvalanche').setLevel(logging.DEBUG)
        log.debug('Debug logging enabled')

    timer = PipelineTimer()
    log.info(f"prepare_dataset arguments: {locals()}")

    # ================================================================
    # Step 1: Validate all input arguments
    # ================================================================
    timer.step('1_validation')

    if avalanche_date is not None:
        avalanche_date = validate_date(avalanche_date)
    aoi = validate_aoi(aoi)

    S1_REVISIT_DAYS = 12
    if start_date is None and avalanche_date is not None:
        start_date = avalanche_date - pd.Timedelta(days=6 * S1_REVISIT_DAYS)
        log.info(f'No start date provided. Using {start_date}')
    if stop_date is None and avalanche_date is not None:
        stop_date = avalanche_date + pd.Timedelta(days=3 * S1_REVISIT_DAYS)
        log.info(f'No stop date provided. Using {stop_date}')

    if start_date is not None and stop_date is not None:
        start_date, stop_date = validate_start_end(start_date, stop_date)

    crs = validate_crs(crs)

    if resolution is None:
        if crs.is_projected:
            resolution = 30
        else:
            resolution = 1 / 3600
        log.info(f'No resolution provided. Using: {resolution}')

    resolution = validate_resolution(resolution)
    cache_dir = validate_path(cache_dir, should_exist=None, make_directory=True)
    log.info('Initial validation checks passed')

    # ================================================================
    # Step 2: Set up cache directory structure
    # ================================================================
    timer.step('2_setup_cache')

    for sub in ['opera', 'arrays']:
        cache_dir.joinpath(sub).mkdir(exist_ok=True)

    if avalanche_date is not None:
        ds_stem = job_name or avalanche_date.strftime('%Y-%m-%d')
    else:
        ds_stem = job_name or 'full_season'
    ds_nc = cache_dir.joinpath(ds_stem).with_suffix('.nc')
    log.info(f'Dataset will be saved to {ds_nc}')

    # ================================================================
    # Step 3: Load or assemble dataset
    # ================================================================
    timer.step('3_load_assemble_dataset')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:
        if not ds_nc.exists() or ds_nc.stat().st_size == 0:
            log.info('Netcdf not found. Assembling dataset now.')
        elif overwrite:
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
    else:
        log.info(f'Found netcdf at {ds_nc}. Loading from cache...')
        ds = load_netcdf_to_dataset(ds_nc)

    # FlowPy runout
    track_gpkg_path = Path(track_gpkg) if track_gpkg else ds_nc.with_suffix('.gpkg')
    _flowpy_vars = ['cell_counts', 'runout_angle', 'release_zones']
    missing_flowpy = not all(v in ds.data_vars for v in _flowpy_vars)
    gpkg_exists = track_gpkg_path.exists() and track_gpkg_path.stat().st_size > 0

    if not missing_flowpy and gpkg_exists and not overwrite:
        log.info('FlowPy outputs already present in dataset.')
    elif gpkg_exists and not overwrite and static_fp is not None and Path(static_fp).exists():
        log.info('Loading flowpy variables from existing netcdf: %s', static_fp)
        donor_ds = load_netcdf_to_dataset(Path(static_fp))
        for v in _flowpy_vars:
            if v in donor_ds.data_vars:
                ds[v] = donor_ds[v]
                ds[v].attrs = donor_ds[v].attrs
        del donor_ds
    else:
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg_path, driver='GPKG')
        log.info(f'Saving netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc, overwrite=True)

    validate_canonical(ds)

    # ================================================================
    # Step 3.5: Preprocessing (TV despeckle)
    # ================================================================
    timer.step('3.5_preprocessing')

    if ds.attrs.get('preprocessed') != 'rtc_tv':
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs['preprocessed'] = 'rtc_tv'
        log.info(f'Saving preprocessed netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc, overwrite=True)

    timer.summary()
    log.info('Dataset preparation complete.')
    return ds, track_gpkg_path


def run_empirical_detection(
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
        temporal_decay_factor=6,
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
    temporal_decay_factor : float, optional
        Exponential decay constant (tau) in days for temporal weighting.
        Larger values give more weight to SAR acquisitions farther from the
        avalanche date. Default 6.
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

        # log.info(f'Saving netcdf to {ds_nc}')
        # export_netcdf(ds, ds_nc)

    else:
        log.info(f'Found netcdf at {ds_nc}. Loading from cache...')
        ds = load_netcdf_to_dataset(ds_nc)

    # add in flowpy outputs and generate track list
    track_gpkg = ds_nc.with_suffix('.gpkg') if track_gpkg is None else track_gpkg
    _flowpy_vars = ['cell_counts', 'runout_angle', 'release_zones']
    missing_flowpy_vars = not all(v in ds.data_vars for v in _flowpy_vars)
    gpkg_exists = track_gpkg.exists() and track_gpkg.stat().st_size > 0

    if not missing_flowpy_vars and gpkg_exists and not overwrite:
        # Everything already in ds — just load the paths GeoDataFrame
        paths_gdf = gpd.read_file(track_gpkg)
    elif gpkg_exists and not overwrite and static_fp is not None and Path(static_fp).exists():
        # Gpkg exists but flowpy vars missing from this date's ds — load them
        # from an existing netcdf (static_fp) instead of re-running flowpy
        log.info('Loading flowpy variables from existing netcdf: %s', static_fp)
        donor_ds = load_netcdf_to_dataset(Path(static_fp))
        for v in _flowpy_vars:
            if v in donor_ds.data_vars:
                ds[v] = donor_ds[v]
                ds[v].attrs = donor_ds[v].attrs
        del donor_ds
        paths_gdf = gpd.read_file(track_gpkg)
    else:
        # No cached flowpy outputs — run from scratch
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg, driver='GPKG')
        log.info(f'Saving netcdf to {ds_nc}')
        export_netcdf(ds, ds_nc, overwrite=True)

    validate_canonical(ds)

    # Early exit: if all pipeline outputs already exist in the cached dataset
    # and the caller didn't request a re-run, skip all computation steps.
    _PIPELINE_OUTPUTS = [
        'detections', 'p_pixelwise', 'p_empirical', 'p_fcf',
        'p_runout', 'p_slope', 'release_zones', 'distance_mahalanobis',
    ]
    if not overwrite and all(v in ds.data_vars for v in _PIPELINE_OUTPUTS):
        log.info("All pipeline outputs present in cached dataset — skipping recomputation")
        timer.summary()
        import gc; gc.collect()
        return ds

    # 3.5 - Preprocessing
    # rtc pre-processing is a homomorphic total variation based despeckling on each time step for each pol
    if ds.attrs.get('preprocessed') != 'rtc_tv':
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs['preprocessed'] = 'rtc_tv'

    # ================================================================
    # Step 4: Calculate spatial domain weights
    # ================================================================
    timer.step('4_calculate_weights')

    # Lazy imports: detection-only dependencies (avoid broken import chain
    # from sarvalanche.ml.inference when only prepare_dataset is needed)
    from sarvalanche.weights.pipelines import get_static_weights
    from sarvalanche.probabilities.pipelines import get_static_probabilities, group_classes
    from sarvalanche.detection.pixelwise import get_pixelwise_probabilities
    from sarvalanche.masks.pipelines import apply_exclusion_masks

    ds = get_static_weights(ds, avalanche_date, temporal_decay_factor=temporal_decay_factor)

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
    from sarvalanche.io.load_data import cleanup_temp_files
    cleanup_temp_files()
    import gc
    gc.collect()

    return ds


def run_ml_detections(
        aoi,
        cache_dir,
        model_weights,
        avalanche_date=None,
        resolution=None,
        crs='EPSG:4326',
        start_date=None,
        stop_date=None,
        static_fp=None,
        track_gpkg=None,
        overwrite=False,
        job_name=None,
        hrrr_fp=None,
        max_span_days=60,
        inference_stride=32,
        inference_batch_size=16,
        onset_threshold=0.2,
        onset_gap_days=18,
        device=None,
        debug=False):
    """
    Run ML-based pairwise debris detection pipeline.

    Reuses the same data assembly and preprocessing as run_detection (steps 1-3.5),
    then runs per-pair CNN inference on pre-denoised backscatter changes and
    aggregates detections via temporal onset.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in WGS84.
    cache_dir : str or Path
        Directory for caching intermediate files and outputs.
    model_weights : str or Path
        Path to trained model checkpoint (.pt).
    avalanche_date : str or datetime, optional
        If provided, only pairs crossing this date are used. If None, all
        pairs with span <= max_span_days are evaluated (full-season mode).
    resolution : float, optional
        Spatial resolution. Defaults to 30m for projected CRS.
    crs : str, optional
        Target CRS. Default 'EPSG:4326'.
    start_date, stop_date : str or datetime, optional
        SAR acquisition window.
    static_fp : Path, optional
        Pre-built static layers NetCDF.
    track_gpkg : Path, optional
        FlowPy track cache.
    overwrite : bool
        Recompute cached results.
    job_name : str, optional
        Output filename stem.
    hrrr_fp : Path, optional
        HRRR temperature NetCDF for melt filtering in temporal onset.
    max_span_days : int
        Maximum pair span in days.
    inference_stride : int
        Sliding window stride (pixels).
    inference_batch_size : int
        Batch size for model inference.
    onset_threshold : float
        Probability threshold for temporal onset.
    onset_gap_days : int
        Temporal gap to separate distinct events.
    device : str, optional
        Torch device ('cpu', 'mps', 'cuda').
    debug : bool
        Enable debug logging.

    Returns
    -------
    xr.Dataset
        Temporal onset results with per-event arrays.
    """
    import xarray as xr

    from sarvalanche.ml.pairwise_debris_classifier.inference import (
        load_model,
        run_all_pairs_inference,
    )
    from sarvalanche.ml.pairwise_debris_classifier.static_stack import build_static_stack
    from sarvalanche.ml.pairwise_debris_classifier.temporal_onset import (
        run_pair_temporal_onset,
    )

    if debug:
        logging.getLogger('sarvalanche').setLevel(logging.DEBUG)

    timer = PipelineTimer()
    model_weights = Path(model_weights)

    # ================================================================
    # Step 1: Validate inputs
    # ================================================================
    timer.step('1_validation')

    if avalanche_date is not None:
        avalanche_date = validate_date(avalanche_date)
    aoi = validate_aoi(aoi)

    S1_REVISIT_DAYS = 12
    if start_date is None and avalanche_date is not None:
        start_date = avalanche_date - pd.Timedelta(days=6 * S1_REVISIT_DAYS)
    if stop_date is None and avalanche_date is not None:
        stop_date = avalanche_date + pd.Timedelta(days=3 * S1_REVISIT_DAYS)

    if start_date is not None and stop_date is not None:
        start_date, stop_date = validate_start_end(start_date, stop_date)

    crs = validate_crs(crs)

    if resolution is None:
        resolution = 30 if crs.is_projected else 1 / 3600
    resolution = validate_resolution(resolution)

    cache_dir = validate_path(cache_dir, should_exist=None, make_directory=True)

    # ================================================================
    # Step 2: Cache setup
    # ================================================================
    timer.step('2_setup_cache')

    for sub in ['opera', 'arrays', 'ml_results']:
        cache_dir.joinpath(sub).mkdir(exist_ok=True)

    ds_stem = job_name or (avalanche_date.strftime('%Y-%m-%d') if avalanche_date else 'full_season')
    ds_nc = cache_dir.joinpath(ds_stem).with_suffix('.nc')

    # ================================================================
    # Step 3: Load or assemble dataset
    # ================================================================
    timer.step('3_load_assemble_dataset')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:
        ds = assemble_dataset(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            resolution=resolution,
            crs=crs,
            cache_dir=cache_dir,
            static_layer_nc=static_fp,
            sar_only=False)
    else:
        log.info('Loading cached dataset: %s', ds_nc)
        ds = load_netcdf_to_dataset(ds_nc)

    # FlowPy runout (needed for cell_counts static channel)
    track_gpkg_path = Path(track_gpkg) if track_gpkg else ds_nc.with_suffix('.gpkg')
    _flowpy_vars = ['cell_counts', 'runout_angle', 'release_zones']
    missing_flowpy = not all(v in ds.data_vars for v in _flowpy_vars)

    if missing_flowpy or (not track_gpkg_path.exists() and overwrite):
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg_path, driver='GPKG')
        export_netcdf(ds, ds_nc, overwrite=True)

    validate_canonical(ds)

    # ================================================================
    # Step 3.5: Preprocessing (TV despeckle each timestep)
    # ================================================================
    timer.step('3.5_preprocessing')

    if ds.attrs.get('preprocessed') != 'rtc_tv':
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs['preprocessed'] = 'rtc_tv'

    # ================================================================
    # Step 4: Build static stack
    # ================================================================
    timer.step('4_static_stack')

    static_scene = build_static_stack(ds)
    log.info('Static stack: %s', static_scene.shape)

    # ================================================================
    # Step 5: Load model
    # ================================================================
    timer.step('5_load_model')

    model, torch_device = load_model(model_weights, device=device)

    # ================================================================
    # Step 6: Per-pair inference
    # ================================================================
    timer.step('6_pair_inference')

    pair_probs, pair_meta = run_all_pairs_inference(
        ds, static_scene, model, torch_device,
        max_span_days=max_span_days,
        stride=inference_stride,
        batch_size=inference_batch_size,
    )

    # ================================================================
    # Step 7: Temporal onset
    # ================================================================
    timer.step('7_temporal_onset')

    hrrr_ds = None
    if hrrr_fp is not None:
        hrrr_fp = Path(hrrr_fp)
        if hrrr_fp.exists():
            hrrr_ds = xr.open_dataset(hrrr_fp)
            log.info('Loaded HRRR: %s', hrrr_fp)

    onset_result, onset_dates, _ = run_pair_temporal_onset(
        pair_probs, pair_meta,
        threshold=onset_threshold,
        gap_days=onset_gap_days,
        hrrr_ds=hrrr_ds,
        coords={'y': ds.y.values, 'x': ds.x.values},
        crs=str(ds.rio.crs) if ds.rio.crs else None,
    )

    # ================================================================
    # Step 8: Export
    # ================================================================
    timer.step('8_export')

    ml_dir = cache_dir.joinpath('ml_results')
    onset_nc = ml_dir / f'{ds_stem}_onset.nc'
    onset_result.to_netcdf(onset_nc)
    log.info('Saved onset results: %s', onset_nc)

    # Export key layers as GeoTIFFs
    for var in ['candidate_mask', 'confidence', 'peak_prob']:
        if var in onset_result:
            tif_path = ml_dir / f'{ds_stem}_{var}.tif'
            onset_result[var].astype(float).rio.to_raster(str(tif_path))
            log.info('Exported %s', tif_path.name)

    timer.summary()

    if hrrr_ds is not None:
        hrrr_ds.close()

    import gc
    gc.collect()

    return onset_result