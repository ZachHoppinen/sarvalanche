
import atexit
import logging
import os
import time
from pathlib import Path
import tempfile

import math

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import py3dep
import pygeohydro as gh
import rasterio
from rasterio.warp import reproject, Resampling
import dask.array as da
import rioxarray
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

tmp_files: list[str] = []

log = logging.getLogger(__name__)

def _make_mmap(shape, dtype, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_files.append(tmp.name)
    return np.memmap(tmp.name, dtype=dtype, mode='w+', shape=shape)


def cleanup_temp_files() -> int:
    """Delete all memmap temp files created by ``_make_mmap``.

    Call this after each pipeline run (once the dataset has been exported
    to netCDF) to avoid accumulating 7-11 GB temp files across runs.

    Returns the number of files removed.
    """
    removed = 0
    while tmp_files:
        fp = tmp_files.pop()
        try:
            os.unlink(fp)
            removed += 1
        except OSError:
            pass
    if removed:
        log.debug("cleanup_temp_files: removed %d temp memmap files", removed)
    return removed


atexit.register(cleanup_temp_files)

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


def _get_hansen_tree_cover(aoi, aoi_crs, ref_grid=None):
    """Fetch Hansen Global Forest Change tree cover (0-100%) for any location.

    Uses COG tiles hosted on Google Cloud Storage.  Tile naming uses the
    upper-left corner at 10° increments, e.g. ``70N_150W``.
    """
    import rioxarray  # noqa: F401
    import math

    # Project AOI to EPSG:4326 to determine which tiles we need
    g = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = g.total_bounds

    # Hansen tiles: upper-left corner, 10° steps
    # lat tiles: 80N, 70N, 60N, ..., 50S  (upper-left latitude)
    # lon tiles: 180W, 170W, ..., 170E    (upper-left longitude)
    def _tile_corner(val, step, is_lat=True):
        """Round up to the next tile boundary."""
        return math.ceil(val / step) * step

    lat_lo = _tile_corner(miny, 10)
    lat_hi = _tile_corner(maxy, 10)
    lon_lo = _tile_corner(minx, -10) if minx < 0 else _tile_corner(minx, 10) - 10
    lon_hi = _tile_corner(maxx, -10) if maxx < 0 else _tile_corner(maxx, 10) - 10

    # Build tile URLs
    base = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
    tiles = []
    for lat in range(lat_lo, lat_hi + 1, 10):
        for lon in range(lon_lo, lon_hi + 1, 10):
            lat_str = f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):03d}{'E' if lon >= 0 else 'W'}"
            url = f"{base}/Hansen_GFC-2023-v1.11_treecover2000_{lat_str}_{lon_str}.tif"
            tiles.append(url)

    log.info("Hansen tree cover: fetching %d tile(s)", len(tiles))

    pieces = []
    for url in tiles:
        try:
            da = rioxarray.open_rasterio(url)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            if "band" in clipped.dims:
                clipped = clipped.isel(band=0, drop=True)
            pieces.append(clipped.astype(np.float32))
        except Exception as e:
            log.warning("Hansen tile %s failed: %s", url, e)

    if not pieces:
        log.warning("No Hansen tree cover data found, returning zeros")
        if ref_grid is not None:
            return xr.zeros_like(ref_grid, dtype=np.float32).assign_attrs(
                units="percent", source="hansen_gfc", product="forest_cover",
            )
        raise ValueError("No Hansen tree cover data and no ref_grid to create zeros")

    if len(pieces) == 1:
        fcf = pieces[0]
    else:
        from rioxarray.merge import merge_arrays
        fcf = merge_arrays(pieces)

    if ref_grid is not None:
        fcf = fcf.rio.reproject_match(ref_grid)

    return fcf.assign_attrs(units="percent", source="hansen_gfc", product="forest_cover")


def get_forest_cover(aoi, aoi_crs, ref_grid=None):
    """Get forest cover fraction (0-100%).

    Tries NLCD first (CONUS). If the result is all-NaN (e.g. Alaska),
    falls back to Hansen Global Forest Change tree cover.
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    try:
        fcf = _with_retry(
            lambda: gh.nlcd_bygeom(geometry=g)[0]["canopy_2021"],
            label="nlcd_bygeom(canopy_2021)",
        )
        # Check if NLCD actually has data (Alaska returns all-NaN)
        if np.all(np.isnan(fcf.values)):
            log.info("NLCD canopy is all-NaN (outside CONUS?), falling back to Hansen GFC")
            raise ValueError("NLCD no data")
        if ref_grid is not None:
            fcf = fcf.rio.reproject_match(ref_grid)
        return fcf.assign_attrs(units="percent", source="nlcd", product="forest_cover")
    except Exception:
        log.info("Using Hansen Global Forest Change tree cover")
        return _get_hansen_tree_cover(aoi, aoi_crs, ref_grid)

def _esa_worldcover_tile_url(lat, lon):
    """Build ESA WorldCover 2021 v200 tile URL for a given lat/lon.

    Tiles are 3x3 degree, named by their SW corner.
    """
    lat_tile = int(math.floor(lat / 3) * 3)
    lon_tile = int(math.floor(lon / 3) * 3)
    lat_str = f"N{abs(lat_tile):02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
    lon_str = f"E{abs(lon_tile):03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
    return (
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
    )


def _get_esa_water(aoi, aoi_crs, ref_grid=None):
    """Get water extent from ESA WorldCover 2021 (class 80 = permanent water).

    Global coverage including Alaska. Used as fallback when NLCD is unavailable.
    """
    bounds = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds

    urls = set()
    for lat in [miny, maxy]:
        for lon in [minx, maxx]:
            urls.add(_esa_worldcover_tile_url(lat, lon))

    log.info("ESA WorldCover water: fetching %d tile(s)", len(urls))

    pieces = []
    for url in urls:
        try:
            da = rioxarray.open_rasterio(url)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            if "band" in clipped.dims:
                clipped = clipped.isel(band=0, drop=True)
            pieces.append(clipped)
        except Exception as e:
            log.warning("ESA WorldCover tile %s failed: %s", url, e)

    if not pieces:
        log.warning("No ESA WorldCover data found, returning zeros")
        if ref_grid is not None:
            return xr.zeros_like(ref_grid, dtype=np.int32).assign_attrs(
                units="binary", source="esa_worldcover", product="water_extent",
                description="1=water, 0=land",
            )
        raise ValueError("No ESA WorldCover data and no ref_grid to create zeros")

    if len(pieces) == 1:
        lc = pieces[0]
    else:
        from rioxarray.merge import merge_arrays
        lc = merge_arrays(pieces)

    water_extent = (lc == 80).astype(int)

    if ref_grid is not None:
        water_extent = water_extent.rio.reproject_match(ref_grid)

    return water_extent.assign_attrs(
        units="binary",
        source="esa_worldcover",
        product="water_extent",
        description="1=water, 0=land",
    )


def get_water_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """Get water extent mask (1=water, 0=land).

    Tries NLCD first (CONUS). If the result is all-NaN (e.g. Alaska),
    falls back to ESA WorldCover 2021 (global coverage).
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    try:
        years = {'impervious': [year], 'cover': [year], 'canopy': [year], 'descriptor': [year]}
        lc = gh.nlcd_bygeom(geometry=g, years=years)[0][f"cover_{year}"]
        if "band" in lc.dims:
            lc = lc.isel(band=0, drop=True)
        if np.all(np.isnan(lc.values)):
            log.info("NLCD cover is all-NaN (outside CONUS?), falling back to ESA WorldCover")
            raise ValueError("NLCD no data")
        water_extent = (lc == 11).astype(int)
        if ref_grid is not None:
            water_extent = water_extent.rio.reproject_match(ref_grid)
        return water_extent.assign_attrs(
            units="binary", source="nlcd", product="water_extent",
            description="1=water, 0=land",
        )
    except Exception:
        log.info("Using ESA WorldCover for water extent")
        return _get_esa_water(aoi, aoi_crs, ref_grid)


def get_urban_extent(aoi, aoi_crs, ref_grid=None, year=2021):
    """Get urban/developed extent mask (1=developed, 0=undeveloped).

    Tries NLCD first (CONUS). If the result is all-NaN (e.g. Alaska),
    falls back to ESA WorldCover 2021 (class 50 = built-up).
    """
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    try:
        years = {'impervious': [year], 'cover': [year], 'canopy': [year], 'descriptor': [year]}
        lc = gh.nlcd_bygeom(geometry=g, years=years)[0][f"cover_{year}"]
        if "band" in lc.dims:
            lc = lc.isel(band=0, drop=True)
        if np.all(np.isnan(lc.values)):
            log.info("NLCD cover is all-NaN (outside CONUS?), falling back to ESA WorldCover")
            raise ValueError("NLCD no data")
        urban_extent = ((lc >= 21) & (lc <= 24)).astype(int)
        if ref_grid is not None:
            urban_extent = urban_extent.rio.reproject_match(ref_grid)
        return urban_extent.assign_attrs(
            units="binary", source="nlcd", product="urban_extent",
            description="1=developed, 0=undeveloped",
        )
    except Exception:
        log.info("Using ESA WorldCover for urban extent")
        return _get_esa_urban(aoi, aoi_crs, ref_grid)


def _get_esa_urban(aoi, aoi_crs, ref_grid=None):
    """Get urban extent from ESA WorldCover 2021 (class 50 = built-up)."""
    bounds = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds

    urls = set()
    for lat in [miny, maxy]:
        for lon in [minx, maxx]:
            urls.add(_esa_worldcover_tile_url(lat, lon))

    log.info("ESA WorldCover urban: fetching %d tile(s)", len(urls))

    pieces = []
    for url in urls:
        try:
            da = rioxarray.open_rasterio(url)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            if "band" in clipped.dims:
                clipped = clipped.isel(band=0, drop=True)
            pieces.append(clipped)
        except Exception as e:
            log.warning("ESA WorldCover tile %s failed: %s", url, e)

    if not pieces:
        log.warning("No ESA WorldCover data found, returning zeros")
        if ref_grid is not None:
            return xr.zeros_like(ref_grid, dtype=np.int32).assign_attrs(
                units="binary", source="esa_worldcover", product="urban_extent",
                description="1=developed, 0=undeveloped",
            )
        raise ValueError("No ESA WorldCover data and no ref_grid to create zeros")

    if len(pieces) == 1:
        lc = pieces[0]
    else:
        from rioxarray.merge import merge_arrays
        lc = merge_arrays(pieces)

    urban_extent = (lc == 50).astype(int)

    if ref_grid is not None:
        urban_extent = urban_extent.rio.reproject_match(ref_grid)

    return urban_extent.assign_attrs(
        units="binary",
        source="esa_worldcover",
        product="urban_extent",
        description="1=developed, 0=undeveloped",
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