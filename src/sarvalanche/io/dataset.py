from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import logging

from sarvalanche.utils.validation import validate_dates, validate_aoi, validate_crs, validate_resolution, validate_canonical

from sarvalanche.utils.grid import make_reference_grid

from sarvalanche.io.find_data import find_asf_urls, find_earthaccess_urls
from sarvalanche.utils.download import download_urls_parallel

from sarvalanche.utils.constants import RTC_FILETYPES, SENTINEL1
from asf_search.constants import RTC, RTC_STATIC

from sarvalanche.io.load_data import load_reproject_concat_rtc, get_dem, get_forest_cover, get_slope
from sarvalanche.utils.raster_utils import combine_close_images

from sarvalanche.masks.debris_flow_modeling import generate_runcount_alpha_angle

log = logging.getLogger(__name__)

def assemble_dataset(
    aoi,
    start_date=None,
    stop_date=None,
    crs=None,
    resolution=None,
    cache_dir=Path("/tmp/sarvalanche_cache"),
) -> xr.Dataset:
    """
    Assemble SAR + auxiliary datasets for avalanche detection.

    Returns
    -------
    xr.Dataset
        Contains SAR backscatter, incidence angle, DEM, slope, forest cover, etc.
    """
    # --- 1. Validate inputs ---
    start_date, stop_date = validate_dates(start_date, stop_date)
    aoi = validate_aoi(aoi)
    crs = validate_crs(crs)
    resolution = validate_resolution(resolution, crs=crs)

    # --- 2. Create reference grid ---
    ref_grid = make_reference_grid(aoi=aoi, crs=crs, resolution=resolution)

    # --- 3. Find and download ASF RTC data ---
    urls = find_asf_urls(aoi, start_date, stop_date, product_type=RTC)
    fps = download_urls_parallel(urls, cache_dir.joinpath('opera'))

    # --- 4. Load & merge backscatter by file type ---
    ds = xr.Dataset()
    for filetype in RTC_FILETYPES:
        subtype_files = [f for f in fps if f.stem.endswith(filetype)]
        da = load_reproject_concat_rtc(subtype_files, ref_grid, filetype)
        da = combine_close_images(da.sortby("time"))
        da.attrs = {'units': 'linear', 'source': SENTINEL1, 'product': RTC}
        ds[filetype] = da

    # --- 5. Mask ---
    ds["VV"] = ds["VV"].where(ds["mask"] == 0)
    ds["VH"] = ds["VH"].where(ds["mask"] == 0)
    ds = ds.rename({"mask": "lia_mask"})

    # --- 6. Load static LIA ---
    lia_urls = find_asf_urls(aoi, start_date = None, stop_date = None, product_type=RTC_STATIC)
    lia_fps = download_urls_parallel(lia_urls, cache_dir.joinpath('opera'))
    lia = load_reproject_concat_rtc(lia_fps, ref_grid, "lia")

    def combine_track(track_da):
        return track_da.max(dim="time")  # or mean if desired

    ds["lia"] = np.deg2rad(
        lia.groupby("track").apply(combine_track)
    ).rename({"track": "static_track"})

    ds['lia'].attrs = {'units': 'radians', 'source': SENTINEL1, 'product': RTC_STATIC}

    # --- 7. Load auxiliary layers ---
    log.info('Getting DEM')
    ds["dem"] = get_dem(aoi, crs, ref_grid)
    log.info('Getting slope')
    ds["slope"] = get_slope(aoi, crs, ref_grid)
    log.info('Getting fcf')
    ds["fcf"] = get_forest_cover(aoi, crs, ref_grid)

    ds = generate_runcount_alpha_angle(ds)

    ds['time'] = pd.to_datetime(ds['time']).tz_localize(None)

    validate_canonical(ds, require_time= None)

    return ds
