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

from sarvalanche.io.find_urls import find_asf_urls
from sarvalanche.io.load_datatypes import load_reproject_concat_rtc
from sarvalanche.utils import combine_close_images

from sarvalanche.utils import download_urls_parallel
from sarvalanche.utils.constants import RTC_FILETYPES
from sarvalanche.utils import combine_close_images

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

    # ------------- Reference grid ------------- #
    # make reference grid for all other data products
    ref_grid = make_reference_grid(aoi = aoi, crs = crs, resolution = resolution)

    # ------------- Load S1 Data ------------- #
    urls = find_asf_urls(aoi, start_date, stop_date, product_type=RTC)
    fps = download_urls_parallel(urls, cache_dir)


    ds = xr.Dataset()
    for filetype in RTC_FILETYPES:
        subtype_files = [f for f in fps if f.stem.endswith(filetype)]
        da = load_reproject_concat_rtc(subtype_files, ref_grid, filetype)
        da = combine_close_images(da.sortby('time'))
        ds[filetype] = da

    # validate canonical shape (time, y, x) with required attributes
    validate_canonical(ds)
    # dataset of VV, VH, mask. We mask now
    ds['VV'] = ds['VV'].where(ds['mask'] == 0)
    ds['VH'] = ds['VH'].where(ds['mask'] == 0)
    ds = ds.rename({'mask': 'lia_mask'})

    # grab local incidence angle for each track
    lia_urls = find_asf_urls(aoi, start_date = None, stop_date = None, product_type=asf.PRODUCT_TYPE.RTC_STATIC)
    lia_fps = download_urls_parallel(lia_urls, cache_dir)
    lia = load_reproject_concat_rtc(lia_fps, ref_grid, 'lia')
    def combine_track(track_da):
        # collapse the 'time' dimension using first non-NaN values
        return track_da.max(dim="time")  # or .mean(dim="time") depending on logic
    ds['lia'] = lia.groupby('track').apply(combine_track)

    # ------------- Load ancillary data ------------- #
    ds['dem'] = py3dep.get_dem(geometry = aoi, resolution = 10, crs = crs).rio.reproject_match(ref_grid)

    aspect = py3dep.get_map(layers = 'Aspect Degrees', resolution= 10, geometry = aoi)
    aspect = aspect.where(aspect != aspect.rio.nodata)
    ds['aspect'] = aspect.astype(float).rio.write_nodata(np.nan).rio.reproject_match(ds)

    slope = py3dep.get_map(layers = 'Slope Degrees', resolution= 10, geometry = aoi)
    slope = slope.where(slope != slope.rio.nodata)
    ds['slope'] = slope.astype(float).rio.write_nodata(np.nan).rio.reproject_match(ds)

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