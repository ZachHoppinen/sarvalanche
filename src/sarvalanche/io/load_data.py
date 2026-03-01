
import atexit
import logging
import os
import time
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import py3dep
import pygeohydro as gh
import rasterio
from rasterio.warp import reproject, Resampling
import dask.array as da
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm


from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

import warnings
from rasterio.errors import NotGeoreferencedWarning
# silence NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _with_retry(fn, retries=4, initial_wait=3.0, label=""):
    """Call fn up to `retries` times with exponential backoff.

    Waits 3, 6, 12 seconds between attempts (default).  Logs a warning on
    each failure so transient service outages are visible in the pipeline log.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = initial_wait * (2 ** attempt)
            log.warning(
                "%s: attempt %d/%d failed (%s: %s). Retrying in %.0fs...",
                label, attempt + 1, retries, type(e).__name__, e, wait,
            )
            time.sleep(wait)


def _slope_from_dem(dem: xr.DataArray) -> xr.DataArray:
    """Compute slope in radians from a DEM DataArray using numpy gradient."""
    transform = dem.rio.transform()
    dx, dy = abs(transform[0]), abs(transform[4])
    values = dem.compute().values if hasattr(dem.data, 'compute') else dem.values
    gy, gx = np.gradient(values.astype(float), dy, dx)
    out = xr.DataArray(
        np.arctan(np.sqrt(gx ** 2 + gy ** 2)),
        coords=dem.coords, dims=dem.dims,
    )
    if dem.rio.crs is not None:
        out = out.rio.write_crs(dem.rio.crs)
    return out


def _aspect_from_dem(dem: xr.DataArray) -> xr.DataArray:
    """Compute aspect in radians from a DEM DataArray using numpy gradient."""
    transform = dem.rio.transform()
    dx, dy = abs(transform[0]), abs(transform[4])
    values = dem.compute().values if hasattr(dem.data, 'compute') else dem.values
    gy, gx = np.gradient(values.astype(float), dy, dx)
    out = xr.DataArray(
        np.arctan2(-gy, gx),
        coords=dem.coords, dims=dem.dims,
    )
    if dem.rio.crs is not None:
        out = out.rio.write_crs(dem.rio.crs)
    return out


ALLOWED_EXTENSIONS = (".tif", ".tiff")

RTC_RAW_TO_CANONICAL = {
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "units",
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "backscatter_type",
    "RADAR_BAND": "band",
    "TRACK_NUMBER": "track",
    "ORBIT_PASS_DIRECTION": "direction",
    "PLATFORM": "platform",
}

tmp_files = []

def _make_mmap(shape, dtype, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_files.append(tmp.name)
    return np.memmap(tmp.name, dtype=dtype, mode='w+', shape=shape)

atexit.register(lambda: [os.unlink(f) for f in tmp_files if os.path.exists(f)])

def read_rtc_attrs(fp):
    attrs = {}
    with rasterio.open(fp) as src:
        tags = src.tags()
        attrs['units'] = tags.get("PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION")
        attrs['backscatter_type'] = tags.get("PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION")
        attrs['band'] = tags.get("RADAR_BAND")
        attrs['track'] = tags.get("TRACK_NUMBER")
        attrs['direction'] = tags.get("ORBIT_PASS_DIRECTION")
        attrs['platform'] = tags.get("PLATFORM")
        attrs['time'] = pd.to_datetime(tags.get("ZERO_DOPPLER_START_TIME"))
    return attrs

def load_reproject_concat_rtc(fps, ref_grid, pol, chunks):
    attributes = [read_rtc_attrs(fp) for fp in fps]
    times = [a['time'] for a in attributes]
    tracks = [int(a['track']) for a in attributes]
    directions = [a['direction'] for a in attributes]
    platforms = [a['platform'] for a in attributes]

    with rasterio.open(fps[0]) as src:
        dtype = 'float32'
        nodata = src.nodata

    ny, nx = len(ref_grid.y), len(ref_grid.x)
    nt = len(fps)

    # --- memmap: full array on disk, filled slice by slice ---
    mmap = _make_mmap(shape=(nt, ny, nx), dtype = dtype, suffix = f'_{pol}.dat')
    # tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{pol}.dat')
    # mmap = np.memmap(tmp.name, dtype=dtype, mode='w+', shape=(nt, ny, nx))

    dst_transform = ref_grid.rio.transform()
    dst_crs = ref_grid.rio.crs
    dst_shape = (ny, nx)

    def reproject_one(fp_idx):
        fp, idx = fp_idx
        with rasterio.open(fp) as src:
            img = src.read(1).astype(dtype)
            dst = np.full(dst_shape, np.nan, dtype=dtype)
            reproject(
                source=img,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
                src_nodata=nodata,
                dst_nodata=np.nan
            )
        return idx, dst

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(reproject_one, (fp, i)): i for i, fp in enumerate(fps)}
        for fut in tqdm(as_completed(futures), total=nt, desc=f"Reprojecting {pol}"):
            idx, dst = fut.result()
            mmap[idx] = dst
            del dst

    mmap.flush()

    # --- wrap memmap in Dask with spatial chunks ---
    dask_arr = da.from_array(mmap, chunks=chunks)

    coords = {
        "time": times,
        "y": ref_grid.y.values,
        "x": ref_grid.x.values,
        "track": ("time", tracks),
        "direction": ("time", directions),
        "platform": ("time", platforms),
    }

    out = xr.DataArray(dask_arr, dims=("time", "y", "x"), coords=coords)
    out = out.rio.write_crs(ref_grid.rio.crs)
    out = out.rio.write_transform(dst_transform)
    out = out.rio.write_nodata(np.nan)

    return out

def _clean_and_match(
    da,
    ref_grid,
    *,
    to_radians=False,
):
    da = da.where(da != da.rio.nodata)
    da = da.astype(float).rio.write_nodata(np.nan)
    da = da.rio.reproject_match(ref_grid)

    if to_radians:
        da = np.deg2rad(da)

    return da

def _get_py3dep_map(
    layer,
    aoi,
    aoi_crs,
    ref_grid,
    resolution,
    *,
    to_radians=False,
):
    da = _with_retry(
        lambda: py3dep.get_map(layers=layer, geometry=aoi, crs=aoi_crs, resolution=resolution),
        label=f"py3dep.get_map({layer})",
    )
    return _clean_and_match(da, ref_grid, to_radians=to_radians)


def get_dem(aoi, aoi_crs, ref_grid=None, resolution=30):
    dem = _with_retry(
        lambda: py3dep.get_dem(geometry=aoi, resolution=resolution, crs=aoi_crs),
        label="py3dep.get_dem",
    )
    if ref_grid is not None:
        dem = dem.rio.reproject_match(ref_grid)
    return dem.assign_attrs(units="m", source="py3dep", product="elevation")


def get_slope(aoi, aoi_crs, ref_grid=None, resolution=30, dem=None):
    try:
        slope = _get_py3dep_map(
            layer="Slope Degrees",
            aoi=aoi,
            aoi_crs=aoi_crs,
            ref_grid=ref_grid,
            resolution=resolution,
            to_radians=True,
        )
        if ref_grid is not None:
            slope = slope.rio.reproject_match(ref_grid)
        return slope.assign_attrs(units="radians", source="py3dep", product="slope")
    except Exception as e:
        if dem is None:
            raise
        log.warning("py3dep slope failed after all retries (%s); computing from DEM", e)
        slope = _slope_from_dem(dem)
        if ref_grid is not None:
            slope = slope.rio.reproject_match(ref_grid)
        return slope.assign_attrs(units="radians", source="computed_from_dem", product="slope")


def get_aspect(aoi, aoi_crs, ref_grid=None, resolution=30, dem=None):
    try:
        aspect = _get_py3dep_map(
            layer="Aspect Degrees",
            aoi=aoi,
            aoi_crs=aoi_crs,
            ref_grid=ref_grid,
            resolution=resolution,
            to_radians=True,
        )
        if ref_grid is not None:
            aspect = aspect.rio.reproject_match(ref_grid)
        return aspect.assign_attrs(units="radians", source="py3dep", product="aspect")
    except Exception as e:
        if dem is None:
            raise
        log.warning("py3dep aspect failed after all retries (%s); computing from DEM", e)
        aspect = _aspect_from_dem(dem)
        if ref_grid is not None:
            aspect = aspect.rio.reproject_match(ref_grid)
        return aspect.assign_attrs(units="radians", source="computed_from_dem", product="aspect")


def get_forest_cover(aoi, aoi_crs, ref_grid=None):
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    fcf = _with_retry(
        lambda: gh.nlcd_bygeom(geometry=g)[0]["canopy_2021"],
        label="nlcd_bygeom(canopy_2021)",
    )
    if ref_grid is not None:
        fcf = fcf.rio.reproject_match(ref_grid)
    return fcf.assign_attrs(units="percent", source="nlcd", product="forest_cover")

def get_water_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """
    Get water extent from NLCD land cover classification.

    This identifies permanent water bodies (lakes, rivers, reservoirs)
    from the NLCD land cover product.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest polygon.
    aoi_crs : str or int
        Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    ref_grid : xr.DataArray, optional
        Reference grid to match. If provided, result will be reprojected
        to match this grid's CRS, resolution, and extent.
    year : int, optional
        NLCD year to use. Options: 2001, 2004, 2006, 2008, 2011, 2013,
        2016, 2019, 2021. Default is 2021.

    Returns
    -------
    xr.DataArray
        Binary mask where 1 = water, 0 = not water.

    Notes
    -----
    NLCD water classes:
    - 11: Open Water
    - 12: Perennial Ice/Snow (also masked as water for avalanche context)

    For avalanche detection, we typically want to mask out:
    - Lakes and reservoirs (not avalanche terrain)
    - Large rivers (not avalanche terrain)
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)

    # Get land cover classification
    lc = gh.nlcd_bygeom(geometry=g, years=year)[0][f"cover_{year}"]

    # Create water mask
    # NLCD class 11 = Open Water
    # NLCD class 12 = Perennial Ice/Snow (optional - uncomment if needed)
    water_mask = (lc == 11)  # | (lc == 12)

    # Convert boolean to int (0 or 1)
    water_extent = water_mask.astype(int)

    if ref_grid is not None:
        water_extent = water_extent.rio.reproject_match(ref_grid)

    return water_extent.assign_attrs(
        units="binary",
        source="nlcd",
        product="water_extent",
        description="1=water, 0=land",
        nlcd_classes="11 (Open Water)"
    )


def get_urban_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """
    Get urban/developed extent from NLCD land cover classification.

    This identifies developed areas (cities, towns, roads) which are
    unlikely to have avalanches.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest polygon.
    aoi_crs : str or int
        Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    ref_grid : xr.DataArray, optional
        Reference grid to match. If provided, result will be reprojected
        to match this grid's CRS, resolution, and extent.
    year : int, optional
        NLCD year to use. Options: 2001, 2004, 2006, 2008, 2011, 2013,
        2016, 2019, 2021. Default is 2021.

    Returns
    -------
    xr.DataArray
        Binary mask where 1 = urban/developed, 0 = not developed.

    Notes
    -----
    NLCD developed classes:
    - 21: Developed, Open Space (< 20% impervious)
    - 22: Developed, Low Intensity (20-49% impervious)
    - 23: Developed, Medium Intensity (50-79% impervious)
    - 24: Developed, High Intensity (80-100% impervious)

    For avalanche detection, developed areas should be masked out as they
    are not natural avalanche terrain.

    Alternative: You can also use NLCD impervious surface product for a
    continuous measure of development (0-100% impervious).
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)

    # Get land cover classification
    lc = gh.nlcd_bygeom(geometry=g, years=year)[0][f"cover_{year}"]

    # Create urban mask
    # NLCD classes 21-24 = Developed areas
    urban_mask = (lc >= 21) & (lc <= 24)

    # Convert boolean to int (0 or 1)
    urban_extent = urban_mask.astype(int)

    if ref_grid is not None:
        urban_extent = urban_extent.rio.reproject_match(ref_grid)

    return urban_extent.assign_attrs(
        units="binary",
        source="nlcd",
        product="urban_extent",
        description="1=developed, 0=undeveloped",
        nlcd_classes="21-24 (Developed)"
    )


def get_water_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """
    Get water extent from NLCD land cover classification.

    This identifies permanent water bodies (lakes, rivers, reservoirs)
    from the NLCD land cover product.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest polygon.
    aoi_crs : str or int
        Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    ref_grid : xr.DataArray, optional
        Reference grid to match. If provided, result will be reprojected
        to match this grid's CRS, resolution, and extent.
    year : int, optional
        NLCD year to use. Options: 2001, 2004, 2006, 2008, 2011, 2013,
        2016, 2019, 2021. Default is 2021.

    Returns
    -------
    xr.DataArray
        Binary mask where 1 = water, 0 = not water.

    Notes
    -----
    NLCD water classes:
    - 11: Open Water
    - 12: Perennial Ice/Snow (also masked as water for avalanche context)

    For avalanche detection, we typically want to mask out:
    - Lakes and reservoirs (not avalanche terrain)
    - Large rivers (not avalanche terrain)
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)

    # Get land cover classification
    years = {'impervious': [year], 'cover': [year], 'canopy': [year], 'descriptor': [year]}
    lc = gh.nlcd_bygeom(geometry=g, years=years)[0][f"cover_{year}"]

    # Create water mask
    # NLCD class 11 = Open Water
    # NLCD class 12 = Perennial Ice/Snow (optional - uncomment if needed)
    water_mask = (lc == 11)  # | (lc == 12)

    # Convert boolean to int (0 or 1)
    water_extent = water_mask.astype(int)

    if ref_grid is not None:
        water_extent = water_extent.rio.reproject_match(ref_grid)

    return water_extent.assign_attrs(
        units="binary",
        source="nlcd",
        product="water_extent",
        description="1=water, 0=land",
        nlcd_classes="11 (Open Water)"
    )


def get_urban_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """
    Get urban/developed extent from NLCD land cover classification.

    This identifies developed areas (cities, towns, roads) which are
    unlikely to have avalanches.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest polygon.
    aoi_crs : str or int
        Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    ref_grid : xr.DataArray, optional
        Reference grid to match. If provided, result will be reprojected
        to match this grid's CRS, resolution, and extent.
    year : int, optional
        NLCD year to use. Options: 2001, 2004, 2006, 2008, 2011, 2013,
        2016, 2019, 2021. Default is 2021.

    Returns
    -------
    xr.DataArray
        Binary mask where 1 = urban/developed, 0 = not developed.

    Notes
    -----
    NLCD developed classes:
    - 21: Developed, Open Space (< 20% impervious)
    - 22: Developed, Low Intensity (20-49% impervious)
    - 23: Developed, Medium Intensity (50-79% impervious)
    - 24: Developed, High Intensity (80-100% impervious)

    For avalanche detection, developed areas should be masked out as they
    are not natural avalanche terrain.

    Alternative: You can also use NLCD impervious surface product for a
    continuous measure of development (0-100% impervious).
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)

    # Get land cover classification
    years = {'impervious': [year], 'cover': [year], 'canopy': [year], 'descriptor': [year]}
    lc = gh.nlcd_bygeom(geometry=g, years=years)[0][f"cover_{year}"]

    # Create urban mask
    # NLCD classes 21-24 = Developed areas
    urban_mask = (lc >= 21) & (lc <= 24)

    # Convert boolean to int (0 or 1)
    urban_extent = urban_mask.astype(int)

    if ref_grid is not None:
        urban_extent = urban_extent.rio.reproject_match(ref_grid)

    return urban_extent.assign_attrs(
        units="binary",
        source="nlcd",
        product="urban_extent",
        description="1=developed, 0=undeveloped",
        nlcd_classes="21-24 (Developed)"
    )


def get_impervious_surface(aoi, aoi_crs, ref_grid=None, year=2021):
    """
    Get impervious surface percentage from NLCD.

    This is an alternative to the binary urban mask - it provides a
    continuous measure of development (0-100% impervious surface).

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest polygon.
    aoi_crs : str or int
        Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    ref_grid : xr.DataArray, optional
        Reference grid to match. If provided, result will be reprojected
        to match this grid's CRS, resolution, and extent.
    year : int, optional
        NLCD year to use. Default is 2021.

    Returns
    -------
    xr.DataArray
        Impervious surface as percentage (0-100).

    Notes
    -----
    Impervious surface indicates developed areas:
    - 0% = No development (natural)
    - 1-20% = Low development
    - 20-50% = Moderate development
    - 50%+ = High development

    You can threshold this (e.g., > 10%) to create a binary urban mask.
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)

    # Get impervious surface data
    years = {'impervious': [year], 'cover': [year], 'canopy': [year], 'descriptor': [year]}
    imp = gh.nlcd_bygeom(geometry=g, years=years)[0][f"impervious_{year}"]

    if ref_grid is not None:
        imp = imp.rio.reproject_match(ref_grid)

    return imp.assign_attrs(
        units="percent",
        source="nlcd",
        product="impervious_surface",
        description="Percent impervious surface (0-100)"
    )

def open_ucla_snowmodel(fp):
    da = xr.open_dataset(fp)['SWE_Post'].sel(Stats = 2) # SWE data var and median stat is 3rd (2nd with 0 index) (mean is 0)
    da = da.rename({'Longitude': 'x', 'Latitude': 'y', 'Day':'snowmodel_time'}).transpose('snowmodel_time', 'y', 'x')
    wy = int(Path(fp).stem.split('_')[9].strip('WY'))
    days = da['snowmodel_time'].values  # 0,1,2,...
    real_times = pd.to_datetime(f"{wy}-10-01") + pd.to_timedelta(days, unit='D')
    da['snowmodel_time'] = real_times
    da = da.rio.write_crs('EPSG:4326')
    return da

def get_snowmodel(swe_fps, start_date = None, stop_date = None, ref_grid = None):
    dataarray_list= [open_ucla_snowmodel(fp) for fp in swe_fps]

    swe = xr.combine_by_coords(dataarray_list)
    if isinstance(swe, xr.Dataset):
        swe = swe.to_array().squeeze()


    if start_date is not None or stop_date is not None:
        swe = swe.sel(snowmodel_time = slice(start_date, stop_date))
    if ref_grid is not None: swe = swe.rio.reproject_match(ref_grid)

    return swe.assign_attrs(units="m", source="ucla", product = "swe")


