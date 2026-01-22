from typing import Union
from pathlib import Path
from datetime import datetime

import xarray as xr
from shapely.geometry import Polygon

from io import find_data, load_data
from preprocessing import preprocess_sar_data
from features import get_sar_avalanche_features
from detection import detect_avalanches
from masks import apply_all_masks
from products import generate_output_detections

def run_detection(
    aoi: Polygon,
    start_date: datetime,
    end_date: datetime,
    *,
    sensor: str = "auto",
    masks: dict | None = None,
    dem: Path | None = None,
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
    masks : dict, optional
        Precomputed masks (slope, layover, forest).
    dem : Path, optional
        Path to DEM file for terrain masking. Otherwise downloads automatically.
    detection_params : dict, optional
        Algorithm thresholds and options.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, y, x) containing detection masks
        and optionally intermediate features.
    """

    # -------------------------------------------------------------
    # 1️⃣ Find SAR data
    # -------------------------------------------------------------
    sar_files = find_data(aoi, start_date, end_date, sensor=sensor)
    if not sar_files:
        raise RuntimeError(f"No SAR data found for AOI {aoi.bounds} between {start_date} and {end_date}")

    # -------------------------------------------------------------
    # 2️⃣ Load SAR stack
    # -------------------------------------------------------------
    sar_stack = load_data(sar_files)

    # -------------------------------------------------------------
    # 3️⃣ Preprocess SAR data
    # -------------------------------------------------------------
    preprocessed_stack = preprocess_sar_data(sar_stack)

    # -------------------------------------------------------------
    # 4️⃣ Load DEM if not provided (for terrain masking)
    # -------------------------------------------------------------
    dem_data = dem
    if dem is None:
        dem_data = load_data(aoi, data_type="DEM")  # or custom DEM loader

    # -------------------------------------------------------------
    # 5️⃣ Compute SAR avalanche features
    # -------------------------------------------------------------
    # Returns a dict of features (backscatter, coherence, terrain masks)
    features_dict = get_sar_avalanche_features(
        preprocessed_stack,
        dem=dem_data,
        masks=masks
    )
    backscatter = features_dict["backscatter"]
    coherence = features_dict["coherence"]
    terrain_masks = features_dict.get("terrain_masks", None)

    # -------------------------------------------------------------
    # 6️⃣ Apply masks (terrain, LIA, layover/shadow)
    # -------------------------------------------------------------
    masked_backscatter, masked_coherence = apply_all_masks(
        backscatter,
        coherence,
        terrain_masks=terrain_masks,
        additional_masks=masks
    )

    # -------------------------------------------------------------
    # 7️⃣ Detect avalanches
    # -------------------------------------------------------------
    debris_mask = detect_avalanches(
        masked_backscatter,
        masked_coherence,
        detection_params=detection_params
    )

    # -------------------------------------------------------------
    # 8️⃣ Generate output products
    # -------------------------------------------------------------
    ds = generate_output_detections(
        debris_mask,
        features_dict,
        aoi=aoi
    )

    # -------------------------------------------------------------
    # 9️⃣ Return canonical xarray dataset
    # -------------------------------------------------------------
    return ds