
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

import rasterio
from rasterio.warp import reproject, Resampling
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
                             desc=f"Reprojecting + inserting {pol}"):
            out[idx, :, :] = dst

    return out

# old slower code #
# def _apply_canonical_attrs(da, sensor, product, RAW_TO_CANONICAL) -> xr.DataArray:
#     """Map raw attributes to canonical form"""
#     da.attrs["sensor"] = sensor
#     da.attrs["product"] = product

#     for raw_attr, canonical in RAW_TO_CANONICAL.items():
#         da.attrs[canonical] = da.attrs.get(raw_attr, None)

#     return da

# def _parse_rtc_timestamp(path: Path) -> pd.Timestamp | None:
#     """Try multiple ways to extract acquisition datetime"""
#     stem = path.stem

#     # 1. Standard filename split
#     try:
#         dt = stem.split("_")[4]
#         return pd.to_datetime(dt, format="%Y%m%dT%H%M%SZ")
#     except (IndexError, ValueError):
#         pass

# def load_s1_rtc(path):
#     if path.suffix.lower() not in ALLOWED_EXTENSIONS:
#         raise ValueError(f"Unsupported file type: {path}")

#     da = xr.open_dataarray(path).squeeze('band', drop = True)

#     # Apply canonical attributes
#     da = _apply_canonical_attrs(da, sensor = SENTINEL1, product= OPERA_RTC, RAW_TO_CANONICAL=RTC_RAW_TO_CANONICAL)

#     t = _parse_rtc_timestamp(path)
#     da = da.expand_dims(time=[t])
#     for coord in ("track", "direction", "platform"):
#         da = da.assign_coords({coord: ("time", [da.attrs.get(coord, "unknown")])})

#     return da
