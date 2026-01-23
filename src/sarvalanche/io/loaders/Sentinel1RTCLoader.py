# sarvalanche/io/loader/sentinel1.py

import rioxarray as rxa
import xarray as xr
import numpy as np
from pathlib import Path

from .BaseLoader import BaseLoader
from sarvalanche.utils import download_urls_parallel

class Sentinel1RTCLoader(BaseLoader):
    sensor = "Sentinel-1"
    product = "RTC"

    def _parse_time(self, filename: str):
        """
        OPERA RTC filename → acquisition time
        """
        # example: *_20200101T161622Z_*
        import re, pandas as pd

        match = re.search(r"_(\d{8}T\d{6})Z_", filename)
        if not match:
            raise ValueError(f"Cannot parse time from {filename}")

        return pd.to_datetime(match.group(1))

    def  _open_file(self, path):
        return super()._open_file(path)