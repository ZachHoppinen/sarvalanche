"""Check OPERA RTC-S1 temporal gaps per track across multiple winters in Turnagain Pass."""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from shapely.geometry import box
from sarvalanche.io.find_data import find_asf_urls
from asf_search.constants import RTC

aoi = box(-149.42, 60.56, -148.79, 61.05)

seasons = [
    ("2021-11-01", "2022-03-31"),
    ("2022-11-01", "2023-03-31"),
    ("2023-11-01", "2024-03-31"),
    ("2024-11-01", "2025-03-31"),
]

for season_start, season_end in seasons:
    urls = find_asf_urls(aoi, season_start, season_end, product_type=RTC)

    track_dates = defaultdict(set)
    for url in urls:
        name = str(url).split("/")[-1] if "/" in str(url) else str(url)
        parts = name.split("_")
        track = None
        date = None
        for part in parts:
            if part.startswith("T") and "-" in part:
                track = part.split("-")[0][1:]
            if part.endswith("Z") and len(part) > 10 and part[0] == "2":
                date = part[:8]
        if track and date:
            track_dates[track].add(date)

    print(f"=== {season_start[:4]}-{season_end[:4]} ===")
    for track in sorted(track_dates.keys()):
        dates = sorted(track_dates[track])
        dts = pd.DatetimeIndex([f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates])
        if len(dts) > 1:
            gaps = dts[1:] - dts[:-1]
            max_gap = gaps.max().days
            idx = gaps.argmax()
            gap_start = dts[idx].strftime("%Y-%m-%d")
            gap_end = dts[idx + 1].strftime("%Y-%m-%d")
            # Also count gaps > 24 days
            big_gaps = [(dts[i].strftime("%Y-%m-%d"), dts[i+1].strftime("%Y-%m-%d"), g.days)
                        for i, g in enumerate(gaps) if g.days > 24]
        else:
            max_gap = 0
            gap_start = gap_end = "N/A"
            big_gaps = []

        print(f"  Track {track}: {len(dates)} dates, max gap = {max_gap}d ({gap_start} -> {gap_end})")
        for gs, ge, gd in big_gaps:
            print(f"    GAP: {gs} -> {ge} ({gd}d)")
    print()
    sys.stdout.flush()
