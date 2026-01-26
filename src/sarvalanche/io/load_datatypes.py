
from pathlib import Path
import pandas as pd
import xarray as xr
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

ALLOWED_EXTENSIONS = (".tif", ".tiff")

RTC_RAW_TO_CANONICAL = {
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_EXPRESSION_CONVENTION": "units",
    "PROCESSING_INFORMATION_OUTPUT_BACKSCATTER_NORMALIZATION_CONVENTION": "backscatter_type",
    "RADAR_BAND": "band",
    "TRACK_NUMBER": "track",
    "ORBIT_PASS_DIRECTION": "direction",
    "PLATFORM": "platform",
}

def _apply_canonical_attrs(da, sensor, product, RAW_TO_CANONICAL) -> xr.DataArray:
    """Map raw attributes to canonical form"""
    da.attrs["sensor"] = sensor
    da.attrs["product"] = product

    for raw_attr, canonical in RAW_TO_CANONICAL.items():
        da.attrs[canonical] = da.attrs.get(raw_attr, None)

    return da

def _parse_rtc_timestamp(path: Path) -> pd.Timestamp | None:
    """Try multiple ways to extract acquisition datetime"""
    stem = path.stem

    # 1. Standard filename split
    try:
        dt = stem.split("_")[4]
        return pd.to_datetime(dt, format="%Y%m%dT%H%M%SZ")
    except (IndexError, ValueError):
        pass

def load_s1_rtc(path):
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path}")

    da = xr.open_dataarray(path).squeeze('band', drop = True)

    # Apply canonical attributes
    da = _apply_canonical_attrs(da, sensor = SENTINEL1, product= OPERA_RTC, RAW_TO_CANONICAL=RTC_RAW_TO_CANONICAL)

    t = _parse_rtc_timestamp(path)
    da = da.expand_dims(time=[t])
    for coord in ("track", "direction", "platform"):
        da = da.assign_coords({coord: ("time", [da.attrs.get(coord, "unknown")])})

    return da
