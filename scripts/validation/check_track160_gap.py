"""Check whether Track 160 data is now available from ASF for the Jan-Mar 2025 gap.

Quick diagnostic: searches ASF for OPERA RTC-S1 Track 160 data in the
Turnagain Pass area for the gap period, without downloading anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from shapely.geometry import box
from sarvalanche.io.find_data import find_asf_urls
from asf_search.constants import RTC

# Turnagain Pass approximate bbox (from the season dataset bounds)
aoi = box(-149.42, 60.56, -148.79, 61.05)

# Search just the gap period
start = "2025-01-16"
end = "2025-03-15"

print(f"Searching ASF for OPERA RTC in Turnagain Pass, {start} to {end}...")
urls = find_asf_urls(aoi, start, end, product_type=RTC)

print(f"\nFound {len(urls)} files total")

# Parse track numbers from filenames
from collections import defaultdict
by_track_date = defaultdict(set)
for url in urls:
    name = str(url).split("/")[-1] if "/" in str(url) else str(url)
    # Extract track from T{track}-{burst}
    for part in name.split("_"):
        if part.startswith("T") and "-" in part:
            track = part.split("-")[0][1:]
            # Extract date
            for p2 in name.split("_"):
                if p2.endswith("Z") and len(p2) > 10 and p2[0] == "2":
                    date = p2[:8]
                    by_track_date[track].add(date)
                    break
            break

for track in sorted(by_track_date.keys()):
    dates = sorted(by_track_date[track])
    print(f"\nTrack {track}: {len(dates)} dates")
    for d in dates:
        print(f"  {d[:4]}-{d[4:6]}-{d[6:]}")
