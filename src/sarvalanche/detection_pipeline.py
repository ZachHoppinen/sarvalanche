
import os
from typing import Union
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
import py3dep
import xarray as xr
from shapely.geometry import Polygon
import asf_search as asf
from asf_search.constants import RTC, RTC_STATIC

from sarvalanche.utils.validation import (
    validate_aoi,
    validate_dates,
    validate_crs,
    validate_resolution,
    validate_canonical
)

from sarvalanche.utils.grid import make_reference_grid

from sarvalanche.io.find_data import find_asf_urls
from sarvalanche.io.load_data import load_reproject_concat_rtc
from sarvalanche.utils import combine_close_images

from sarvalanche.utils import download_urls_parallel
from sarvalanche.utils.constants import RTC_FILETYPES
from sarvalanche.utils import combine_close_images


import logging

# Configure logging at the very top of your main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('asf_search').setLevel(logging.WARNING)  # or logging.ERROR
log = logging.getLogger(__name__)


def run_detection(
    aoi: Polygon,
    start_date: datetime,
    stop_date: datetime,
    *,
    cache_dir: Path = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data/opera'),
    masks: dict | None = None,
    dem: Path | None = None,
    detection_params: dict | None = None
) -> xr.Dataset:
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
    sensor : str, optional
        SAR sensor to use ('Sentinel-1', 'NISAR', or 'auto').
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

    # ------------- Validate user inputs ------------- #
    # return pandas datetimes
    start_date, stop_date = validate_dates(start_date, stop_date)
    # returns shapely polygon
    aoi = validate_aoi(aoi)
    # return PyProj CRS
    crs = validate_crs(crs)
    # return tuple of (xres, yres)
    resolution = validate_resolution(resolution, crs = crs)
    os.environ["HYRIVER_CACHE_NAME"] = cache_dir.joinpath('/cache/aiohttp_cache.sqlite')


    # ------------- Reference grid ------------- #
    # make reference grid for all other data products
    ref_grid = make_reference_grid(aoi = aoi, crs = crs, resolution = resolution)

    log.info(f"Arguments: {locals()}")

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

    # -------------------------------------------------------------
    # 7️⃣ Detect avalanches
    # -------------------------------------------------------------
    # debris_mask = detect_avalanches(
        # masked_backscatter,
        # masked_coherence,
        # detection_params=detection_params
    # )

    # -------------------------------------------------------------
    # 8️⃣ Generate output products
    # -------------------------------------------------------------
    # ds = generate_output_detections(
        # debris_mask,
        # features_dict,
        # aoi=aoi
    # )

    # -------------------------------------------------------------
    # 9️⃣ Return canonical xarray dataset
    # -------------------------------------------------------------
    return ds