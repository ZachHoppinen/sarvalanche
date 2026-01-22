# src/sarvalanche/io/finder.py
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon
from typing import List

def find_data(
    aoi: Polygon,
    start_date: datetime,
    end_date: datetime,
    sensor: str = "auto",
    local_archive: Path | None = None
) -> List[Path]:
    """
    Find SAR files for a given AOI and date range.

    Parameters
    ----------
    aoi : Polygon
        Projected-area of interest.
    start_date : datetime
        Start of acquisition range.
    end_date : datetime
        End of acquisition range.
    sensor : str
        SAR sensor ('Sentinel-1', 'NISAR', 'auto').
    local_archive : Path, optional
        Directory to search for pre-downloaded SAR data.

    Returns
    -------
    List[Path]
        List of file paths to SAR data.
    """

    sar_files: List[Path] = []

    # Example: search a local archive if provided
    if local_archive is not None:
        for f in Path(local_archive).rglob("*.h5"):
            # TODO: implement metadata parsing to filter by aoi & date
            sar_files.append(f)

    # TODO: integrate remote search API for Sentinel-1 or NISAR
    # e.g., ASF API or Earthdata

    return sar_files
