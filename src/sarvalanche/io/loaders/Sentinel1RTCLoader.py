# sarvalanche/io/loader/sentinel1.py

import rioxarray as rxa
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import re
import rasterio

from .BaseLoader import BaseLoader
from sarvalanche.utils import download_urls_parallel
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

RTC_TIME_REGEXES = [
    r"(\d{8}T\d{6}Z)",           # 20200101T161622Z
    r"(\d{8}_\d{6})",            # 20200101_161622
    r"(\d{8})",                   # 20200101
]

class Sentinel1RTCLoader(BaseLoader):
    sensor = SENTINEL1
    product = OPERA_RTC

    RAW_TO_CANONICAL = {
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "units",
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "backscatter_type",
    "RADAR_BAND": "band",
    "TRACK_NUMBER": "track",
    "ORBIT_PASS_DIRECTION": "direction",
    "PLATFORM": "platform",
    }

    POLS = ['VV', 'VH', 'mask']

    ALLOWED_EXTENSIONS = (".tif", ".tiff")

    def _parse_time(self, path: Path) -> pd.Timestamp | None:
        stem = path.stem

        # 1. Standard filename split
        try:
            dt = stem.split('_')[4]  # usually the acquisition start
            return pd.to_datetime(dt, format="%Y%m%dT%H%M%SZ")
        except (IndexError, ValueError):
            pass

        # 2. Regex fallback
        for pattern in RTC_TIME_REGEXES:
            m = re.search(pattern, stem)
            if m:
                time_str = m.group(1)
                for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d_%H%M%S", "%Y%m%d"):
                    try:
                        return pd.to_datetime(time_str, format=fmt)
                    except ValueError:
                        continue

        # 3. Rasterio metadata fallback
        try:
            with rasterio.open(path) as src:
                tags = src.tags()
                for key in ("ZERO_DOPPLER_START_TIME",):
                    if key in tags:
                        try:
                            return pd.to_datetime(tags[key])
                        except Exception:
                            continue
        except Exception:
            pass

        # Could not parse time
        raise ValueError(f'Unable to parse time from s1 RTC file: {path}')

    def _apply_canonical_attrs(self, da: xr.DataArray) -> xr.DataArray:
        # Start with sensor and product
        da.attrs["sensor"] = self.sensor
        da.attrs["product"] = self.product
        da.attrs["crs"] = str(da.rio.crs) if da.rio.crs else None

        # Map raw attrs to canonical ones
        for raw_attr, canonical in self.RAW_TO_CANONICAL.items():
            if raw_attr in da.attrs:
                da.attrs[canonical] = da.attrs[raw_attr]
            else:
                # fallback if attribute is missing
                da.attrs[canonical] = None

        return da

    def get_polarization(self, path: Path) -> str:
        name = path.name
        if '_mask.tif' in name: return 'mask'
        elif '_VV.tif' in name: return 'VV'
        elif '_VH.tif' in name: return 'VH'
        raise ValueError(f'Unable to parse polarization from {path}')


    def _open_file(self, path: Path) -> xr.DataArray:
        if not path.suffix.lower() in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Sentinel1RTCLoader only supports GeoTIFF files. Got: {path}"
            )

        da = xr.open_dataarray(path)[0]

        pol = self.get_polarization(path)

        da = da.rename(pol)
        # Apply canonical attributes
        da = self._apply_canonical_attrs(da)

        # Expand time dimension if possible
        t = self._parse_time(path)
        if t is not None:
            da = da.expand_dims(time=[t])
            # assign track/direction/platform as coords along time
            for coord in ("track", "direction", "platform"):
                da = da.assign_coords({coord: ("time", [da.attrs.get(coord, "unknown")])})

        return da
