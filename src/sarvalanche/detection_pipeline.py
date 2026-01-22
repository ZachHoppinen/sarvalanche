from typing import Union
import xarray as xr
from shapely.geometry import Polygon
from datetime import datetime

def run_detection(
    aoi: Polygon,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    *,
    sensor: str = "auto",
    polarization: str = "VV",
    masks: dict | None = None,
    detection_params: dict | None = None
) -> xr.Dataset:
    """
    Run the SARvalanche detection pipeline for a given AOI and date range.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest in projected CRS.
    start_date : str | datetime
        Start of acquisition range.
    end_date : str | datetime
        End of acquisition range.
    sensor : str, optional
        SAR sensor to use ('Sentinel-1', 'NISAR', or 'auto').
    polarization : str, optional
        SAR polarization to use.
    masks : dict, optional
        Precomputed masks (slope, layover, forest).
    detection_params : dict, optional
        Algorithm thresholds and options.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, y, x) containing detection masks
        and optionally intermediate features.
    """