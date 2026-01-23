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

    def  _open_file(self, path):
        da = xr.open_dataarray(path)
