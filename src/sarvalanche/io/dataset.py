from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import logging

from sarvalanche.utils.validation import validate_start_end, validate_aoi, validate_crs, validate_resolution, validate_canonical, validate_path

from sarvalanche.utils.grid import make_reference_grid, make_opera_reference_grid

from sarvalanche.io.find_data import find_asf_urls, find_earthaccess_urls
from sarvalanche.utils.download import download_urls_parallel
from sarvalanche.utils.grid import grids_match

from sarvalanche.utils.constants import RTC_FILETYPES, SENTINEL1
from asf_search.constants import RTC, RTC_STATIC

from sarvalanche.io.load_data import (
    load_reproject_concat_rtc,
    get_dem,
    get_forest_cover,
    get_slope,
    get_aspect,
    get_urban_extent,
    get_water_extent,
    get_snowmodel)
from sarvalanche.utils.raster_utils import combine_close_images

log = logging.getLogger(__name__)

def assemble_dataset(
    aoi,
    start_date=None,
    stop_date=None,
    crs=None,
    resolution=None,
    cache_dir=Path("/tmp/sarvalanche_cache"),
    static_layer_nc = None,
    sar_only = False,
    # TODO implement dask/chunking...
    chunks = {
    'time': 5,      # 5 time steps at once
    'x': 256,       # 256x256 spatial tiles
    'y': 256}

) -> xr.Dataset:
    """
    Assemble SAR + auxiliary datasets for avalanche detection.

    Parameters
    ----------
    chunks : dict, optional
        Dask chunking strategy. Default: {'time': 10, 'x': 512, 'y': 512}
        Adjust based on your dataset size and available memory.

    Returns
    -------
    xr.Dataset
        Contains SAR backscatter, incidence angle, DEM, slope, forest cover, etc.
    """
    # --- 1. Validate inputs ---
    start_date, stop_date = validate_start_end(start_date, stop_date)
    aoi = validate_aoi(aoi)
    crs = validate_crs(crs)
    resolution = validate_resolution(resolution, crs=crs)

    # --- 2. Create reference grid ---
    # TODO convert to opera grid for less reprojecting...
    # ref_grid = make_opera_reference_grid(aoi=aoi, crs=crs)  # would remove resolution. Force to 30 meters...
    ref_grid = make_reference_grid(aoi=aoi, crs=crs, resolution=resolution)

    # --- 3. Find and download ASF RTC data ---
    urls = find_asf_urls(aoi, start_date, stop_date, product_type=RTC)
    fps = download_urls_parallel(urls, cache_dir.joinpath('opera'), description='Downloading S1 RTC')

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
    ds['time'] = pd.to_datetime(ds['time']).tz_localize(None)

    if sar_only:
        return ds

    # get snowmodel
    swe_urls = find_earthaccess_urls(aoi, start_date, stop_date)
    swe_fps = download_urls_parallel(swe_urls, cache_dir.joinpath('snowmodel'), description='Downloading UCLA Snowmodel')
    ds['swe'] = get_snowmodel(swe_fps, start_date, stop_date, ref_grid)

    if static_layer_nc is not None:
        ds = add_static_layers(ds, static_layer_nc)
        return ds

    # --- 6. Load static LIA ---
    static_urls = find_asf_urls(aoi, start_date = None, stop_date = None, product_type=RTC_STATIC)
    static_fps = download_urls_parallel(static_urls, cache_dir.joinpath('opera'), description='Downloading S1 static RTC files')
    lia_fps, anf_fps = [f for f in static_fps if str(f).endswith('local_incidence_angle.tif')], [f for f in static_fps if str(f).endswith('rtc_anf_gamma0_to_beta0.tif')]
    lia = load_reproject_concat_rtc(lia_fps, ref_grid, "lia")
    anf = load_reproject_concat_rtc(anf_fps, ref_grid, "anf")
    # we need to fill nans in anf and lia for weighting...

    def combine_track(track_da):
        return track_da.max(dim="time")  # or mean if desired

    ds["lia"] = np.deg2rad(lia.groupby("track").apply(combine_track)).rename({"track": "static_track"})
    ds["anf"] = anf.groupby("track").apply(combine_track).rename({"track": "static_track"})

    ds['anf'] = ds['anf'].ffill('x').bfill('x').ffill('y').bfill('y')

    ds['lia'].attrs = {'units': 'radians', 'source': SENTINEL1, 'product': RTC_STATIC}
    ds['anf'].attrs = {'units': 'meters', 'source': SENTINEL1, 'product': RTC_STATIC}

    # --- 7. Load auxiliary layers ---
    log.info('Getting DEM')
    ds["dem"] = get_dem(aoi, crs, ref_grid)
    log.info('Getting slope')
    ds["slope"] = get_slope(aoi, crs, ref_grid)
    log.info('Getting aspect')
    ds["aspect"] = get_aspect(aoi, crs, ref_grid)
    log.info('Getting fcf')
    ds["fcf"] = get_forest_cover(aoi, crs, ref_grid)
    log.info('Getting water cover')
    ds["water_mask"] = get_water_extent(aoi, crs, ref_grid)
    log.info('Getting urban')
    ds["urban_mask"] = get_urban_extent(aoi, crs, ref_grid)

    validate_canonical(ds, require_time = None)

    return ds

def load_netcdf_to_dataset(filepath, decode_times=True):
    filepath = validate_path(filepath, should_exist=True)
    assert filepath.suffix == '.nc'

    ds = xr.open_dataset(filepath, decode_times=decode_times)

    if 'crs' in ds.attrs:
        ds = ds.rio.write_crs(ds.attrs['crs'], inplace=True)

    # Drop stray scalar coords that can cause merge conflicts
    scalar_coords_to_drop = [c for c in ds.coords if ds.coords[c].dims == () and c not in ('spatial_ref',)]
    if scalar_coords_to_drop:
        log.debug(f"Dropping scalar coordinates on load: {scalar_coords_to_drop}")
        ds = ds.drop_vars(scalar_coords_to_drop)

    return ds
def add_static_layers(ds: xr.Dataset, static_nc_fp: Path) -> xr.Dataset:
    """
    Load static/auxiliary layers from a NetCDF file and add them to an existing dataset.
    If the CRS or grid of the static file differs from ds, layers will be reprojected to match.

    Parameters
    ----------
    ds : xr.Dataset
        Existing dataset (e.g. from assemble_dataset with sar_only=True).
    static_nc_fp : Path
        Path to a NetCDF file containing static/auxiliary layers.

    Returns
    -------
    xr.Dataset
        Input dataset with static layers merged in.
    """
    static_layers = ["lia", "anf", "dem", "slope", "aspect", "fcf", "water_mask", "urban_mask"]

    static_ds = load_netcdf_to_dataset(static_nc_fp)

    available = [var for var in static_layers if var in static_ds]
    missing = [var for var in static_layers if var not in static_ds]
    if missing:
        raise ValueError(f"The following expected static layers were not found in {static_nc_fp}: {missing}")

    static_subset = static_ds[available]

    if not grids_match(ds, static_subset):
        log.info("CRS or grid mismatch detected — reprojecting static layers to match input dataset.")
        static_subset = static_subset.rio.reproject_match(ds)
    else:
        log.debug("CRS and grid match — skipping reprojection.")

    ds = ds.merge(static_subset)

    return ds