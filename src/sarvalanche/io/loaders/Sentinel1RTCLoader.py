# sarvalanche/io/loader/sentinel1.py

from pathlib import Path
import re

import xarray as xr
import pandas as pd
import rasterio

from .BaseLoader import BaseLoader
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC, RTC_FILETYPES

RTC_TIME_REGEXES = [
    r"(\d{8}T\d{6}Z)",  # 20200101T161622Z
    r"(\d{8}_\d{6})",   # 20200101_161622
    r"(\d{8})",          # 20200101
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

    RTC_FILETYPES = ["VV", "VH", "mask"]
    ALLOWED_EXTENSIONS = (".tif", ".tiff")

    # ---------------- file-specific hooks ----------------

    def _open_file(self, path: Path) -> xr.DataArray:
        """Open a single Sentinel-1 RTC GeoTIFF and return canonical DataArray"""
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path}")

        # Open the first band
        da = xr.open_dataarray(path).squeeze('band', drop = True)

        # Assign polarization as the name
        da = da.rename(self.get_polarization(path))

        # Apply canonical attributes
        da = self._apply_canonical_attrs(da)

        # Add time dimension and track/direction/platform coords
        t = self._parse_time(path)
        if t is not None:
            da = da.expand_dims(time=[t])
            for coord in ("track", "direction", "platform"):
                da = da.assign_coords({coord: ("time", [da.attrs.get(coord, "unknown")])})

        return da

    # ---------------- canonical attributes ----------------

    def _apply_canonical_attrs(self, da: xr.DataArray) -> xr.DataArray:
        """Map raw S1 attributes to canonical form"""
        da.attrs["sensor"] = self.sensor
        da.attrs["product"] = self.product

        for raw_attr, canonical in self.RAW_TO_CANONICAL.items():
            da.attrs[canonical] = da.attrs.get(raw_attr, None)

        return da

    # ---------------- time parsing ----------------

    def _parse_time(self, path: Path) -> pd.Timestamp | None:
        """Try multiple ways to extract acquisition datetime"""
        stem = path.stem

        # 1. Standard filename split
        try:
            dt = stem.split("_")[4]
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

        # Could not parse
        raise ValueError(f"Unable to parse time from {path}")

    # ---------------- polarization ----------------

    def get_polarization(self, path: Path) -> str:
        """Return 'VV', 'VH', or 'mask' from filename"""
        name = path.name
        if "_mask.tif" in name:
            return "mask"
        elif "_VV.tif" in name:
            return "VV"
        elif "_VH.tif" in name:
            return "VH"
        else:
            raise ValueError(f"Cannot determine polarization from {path}")
