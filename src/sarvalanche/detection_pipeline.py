
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr

# io functions
from sarvalanche.utils.projections import resolution_to_degrees
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

# weights and static probabilities
from sarvalanche.weights.pipelines import get_static_weights
from sarvalanche.probabilities.pipelines import get_static_probabilities

# pixelwise terrain, snow, SAR probabilities
from sarvalanche.detection.pixelwise import get_pixelwise_probabilities

# grouping of probabilities
from sarvalanche.probabilities.pipelines import group_classes

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
        overwrite = False,
        job_name = None):
    """
    Run the SARvalanche detection pipeline for a given AOI and date range.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in projected CRS.
    start_date : str | datetime
        Start of acquisition range.
    end_date : str | datetime
        End of acquisition range.
    masks : dict, optional
        Precomputed masks (slope, layover, forest).
    dem : Path, optional
        Path to DEM file for terrain masking. Otherwise downloads automatically.
    detection_params : dict, optional
        Algorithm thresholds and options.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, y, x) containing detection masks
        and optionally intermediate features.
    """
    log.info(f"Arguments: {locals()}")

    log.info('Validating arguments')
    start_date, stop_date = validate_start_end(start_date, stop_date)
    avalanche_date = validate_date(avalanche_date)
    crs = validate_crs(crs)
    resolution = validate_resolution(resolution)
    aoi = validate_aoi(aoi)
    cache_dir = validate_path(cache_dir, should_exist=None, make_directory=True)
    assert overwrite in [True, False]
    log.info(f'Initial validation checks passed')

    for sub_directory_name in ['opera', 'arrays']:
        cache_dir.joinpath(sub_directory_name).mkdir(exist_ok = True)

    ds_stem = f'{avalanche_date.strftime('%Y-%m-%d')}' if job_name is None else f'{job_name}'
    ds_nc = cache_dir.joinpath(ds_stem).with_suffix('.nc')

    log.info(f'Saving results to {ds_nc}')

    if not ds_nc.exists() or ds_nc.stat().st_size == 0 or overwrite:
        if not ds_nc.exists() or ds_nc.stat().st_size == 0: log.info('Netcdf not found. Assembling dataset now.')
        elif overwrite == True: log.info('Netcdf found. Overwriting dataset.')

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
        log.info(f'Found netcdf at {ds_nc}. Loading...')
        ds = load_netcdf_to_dataset(ds_nc)

    validate_canonical(ds)

    # next generate spatial domain weights
    ds = get_static_weights(ds, avalanche_date)

    # generate non-SAR probabilities
    ds = get_static_probabilities(ds, avalanche_date)

    # get pixel-by-pixel probabiliteis based on non-Sar and SAR probalities
    ds['p_pixelwise'] = get_pixelwise_probabilities(ds, avalanche_date)

    # increase likelyhood of neighboring classes
    ds['detections'] = group_classes(ds, cache_dir)

    # -------------------------------------------------------------
    # 8️⃣ Generate output products
    # -------------------------------------------------------------
    validate_canonical(ds)

    log.info(f'Saving final results to {ds_nc}')
    export_netcdf(ds, ds_nc, overwrite = True)

    for var in ['detections', 'p_pixelwise', 'p_empirical']:
        ds[var].astype(float).rio.to_raster(cache_dir.joinpath(f'{var}.tif'))

    # -------------------------------------------------------------
    # 9️⃣ Return canonical xarray dataset
    # -------------------------------------------------------------
    return ds