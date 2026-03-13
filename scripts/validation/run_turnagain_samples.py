"""
Run the empirical detection pipeline for 4 sample dates in Turnagain Pass:
  2 high-danger dates, 2 low-danger dates. tau=6.
Outputs GeoTIFFs to local/cnfaic/sample_detections/<date>/probabilities/
"""

from pathlib import Path
from shapely.geometry import box

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sarvalanche.detection_pipeline import run_detection

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic")
STATIC_NC = BASE / "netcdfs" / "Turnagain_Pass_and_Girdwood" / "season_2024-2025_Turnagain_Pass_and_Girdwood.nc"
TRACK_GPKG = BASE / "netcdfs" / "Turnagain_Pass_and_Girdwood" / "season_2024-2025_Turnagain_Pass_and_Girdwood.gpkg"

# AOI from the season dataset bounds
AOI = box(-149.4236, 60.5576, -148.7897, 61.0537)

DATES = {
    "2024-12-22": "HIGH",
    "2025-01-25": "HIGH",
    "2025-02-09": "LOW",
    "2025-02-13": "LOW",
}

for date_str, label in DATES.items():
    print(f"\n{'='*60}")
    print(f"Running {date_str} ({label} danger), tau=6")
    print(f"{'='*60}")

    cache_dir = BASE / "sample_detections" / f"{date_str}_{label}"

    ds = run_detection(
        aoi=AOI,
        avalanche_date=date_str,
        cache_dir=cache_dir,
        crs="EPSG:4326",
        static_fp=STATIC_NC,
        track_gpkg=TRACK_GPKG,
        temporal_decay_factor=6,
        job_name=f"{date_str}_{label}",
        overwrite=False,
    )
    print(f"Done: {date_str} -> {cache_dir}")
