
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC
import py3dep
import pygeohydro as gh
import rasterio
from rasterio.warp import reproject, Resampling
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import geopandas as gpd

from sarvalanche.io.find_data import find_earthaccess_urls

ALLOWED_EXTENSIONS = (".tif", ".tiff")

RTC_RAW_TO_CANONICAL = {
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "units",
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "backscatter_type",
    "RADAR_BAND": "band",
    "TRACK_NUMBER": "track",
    "ORBIT_PASS_DIRECTION": "direction",
    "PLATFORM": "platform",
}

def preallocate_output(times, y, x, dtype, crs, transform, nodata, time_coords=None):
    """
    Preallocate an xarray.DataArray for output.

    Parameters
    ----------
    times : array-like
        Time coordinates.
    y : array-like
        Y coordinates.
    x : array-like
        X coordinates.
    dtype : numpy dtype
        Data type of the array.
    crs : CRS
        Coordinate reference system (pyproj CRS or string).
    transform : affine.Affine
        Geotransform for rasterio/rioxarray.
    time_coords : dict of array-like, optional
        Optional coordinates to attach along the time dimension.
        Keys are coordinate names, values are arrays of same length as `times`.

    Returns
    -------
    xarray.DataArray
    """
    data = np.full((len(times), len(y), len(x)), nodata, dtype=dtype)

    coords = {
        "time": times,
        "y": y,
        "x": x,
    }

    # Add additional time dimension coordinates if provided
    if time_coords is not None:
        for name, values in time_coords.items():
            if len(values) != len(times):
                raise ValueError(f"Length of time coordinate '{name}' ({len(values)}) does not match length of times ({len(times)})")
            coords[name] = ("time", values)

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords=coords,
    )

    # Attach spatial reference info (requires rioxarray)
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)
    da = da.rio.write_nodata(nodata)
    return da

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


def load_reproject_concat_rtc(fps, ref_grid, pol):
    attributes = [read_rtc_attrs(fp) for fp in fps]
    times = [a['time'] for a in attributes]
    tracks = [int(a['track']) for a in attributes]
    directions = [a['direction'] for a in attributes]
    platforms = [a['platform'] for a in attributes]

    with rasterio.open(fps[0]) as src:
        dtype = src.dtypes[0]
        nodata = src.nodata

    out = preallocate_output(
        times,
        ref_grid.y,
        ref_grid.x,
        dtype,
        ref_grid.rio.crs,
        ref_grid.rio.transform(),
        nodata,
        time_coords={'track': tracks, 'direction': directions, 'platform': platforms}
    )

    dst_transform = ref_grid.rio.transform()
    dst_crs = ref_grid.rio.crs
    dst_shape = (len(ref_grid.y), len(ref_grid.x))
    if dst_transform is None or dst_transform.is_identity:
        raise ValueError("Destination transform is invalid")


    def reproject_one(fp_idx):
        fp, idx = fp_idx

        with rasterio.open(fp) as src:
            if src.transform is None or src.transform.is_identity:
                raise ValueError(f"File is not georeferenced: {fp}")

            if src.crs is None:
                raise ValueError(f"File has no CRS: {fp}")

            img = src.read(1).astype(dtype).astype("float32")
            dst = np.full(dst_shape, np.nan, dtype="float32")

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

    # Parallel reprojection
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=4) as ex:
        for idx, dst in tqdm(ex.map(reproject_one, [(fp, i) for i, fp in enumerate(fps)]),
                             total=len(fps),
                             desc=f"Reprojecting + stacking {pol}"):
            out[idx, :, :] = dst

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
    da = py3dep.get_map(
        layers=layer,
        geometry=aoi,
        crs=aoi_crs,
        resolution=resolution,
    )
    return _clean_and_match(da, ref_grid, to_radians=to_radians)

def get_dem(aoi, aoi_crs, ref_grid = None, resolution=30):
    dem = py3dep.get_dem(
        geometry=aoi,
        resolution=resolution,
        crs=aoi_crs,
    )
    if ref_grid is not None: dem = dem.rio.reproject_match(ref_grid)
    return dem.assign_attrs(units="m", source="py3dep", product = "elevation")

def get_slope(aoi, aoi_crs, ref_grid = None, resolution=30):
    slope = _get_py3dep_map(
        layer="Slope Degrees",
        aoi=aoi,
        aoi_crs=aoi_crs,
        ref_grid=ref_grid,
        resolution=resolution,
        to_radians=True,
    )
    if ref_grid is not None: slope = slope.rio.reproject_match(ref_grid)
    return slope.assign_attrs(units="radians", source="py3dep", product = "slope")

def get_aspect(aoi, aoi_crs, ref_grid = None, resolution=30):
    aspect = _get_py3dep_map(
        layer="Aspect Degrees",
        aoi=aoi,
        aoi_crs=aoi_crs,
        ref_grid=ref_grid,
        resolution=resolution,
        to_radians=True,
    )
    if ref_grid is not None: aspect = aspect.rio.reproject_match(ref_grid)
    return aspect.assign_attrs(units="radians", source="py3dep", product = "aspect")

def get_forest_cover(aoi, aoi_crs, ref_grid = None):
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    fcf = gh.nlcd_bygeom(geometry=g)[0]["canopy_2021"]
    if ref_grid is not None: fcf = fcf.rio.reproject_match(ref_grid)
    return fcf.assign_attrs(units="percent", source="nlcd", product = "forest_cover")

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


