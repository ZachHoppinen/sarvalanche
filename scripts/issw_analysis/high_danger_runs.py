"""
high_danger_runs.py — Identify high-danger periods for SNFAC winters 2021-2025,
fetch observations around each peak, and run sarvalanche for each zone/date.

CLI usage:
    python high_danger_runs.py
    python high_danger_runs.py --center SNFAC --obs-window 6 --peak-height 3.0 --out-dir ./runs
    python high_danger_runs.py --no-fetch   # skip API calls, load cached CSVs

Script / import usage:
    from high_danger_runs import find_high_danger_periods, run_sarvalanche_for_peaks
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path
import gc

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from scipy.signal import find_peaks
from shapely.geometry import box, shape
from tqdm.auto import tqdm
import sys
sys.path.append(str(Path('../data_acquisition').resolve()))
from get_forecast_dangers import get_dangers
from get_avalanche_observations import get_observations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENTER_ID = "SNFAC"

# Winter seasons: Nov 1 → Apr 30
WINTERS = [
    ("2021-11-01", "2022-04-30"),
    ("2022-11-01", "2023-04-30"),
    ("2023-11-01", "2024-04-30"),
    ("2024-11-01", "2025-04-30"),
]

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.avalanche.org",
    "referer": "https://www.avalanche.org/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# ---------------------------------------------------------------------------
# Zone geometry helpers
# ---------------------------------------------------------------------------

def fetch_snfac_zones() -> dict:
    """
    Fetch SNFAC forecast zone polygons from avalanche.org.

    Returns
    -------
    dict: {zone_name: {"geometry": shapely_polygon, "bbox": (minx,miny,maxx,maxy),
                       "bbox_str": "minx,miny,maxx,maxy", "zone_id": ...}}
    """
    url = f"https://api.avalanche.org/v2/public/products/map-layer/{CENTER_ID}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    geojson = r.json()

    zones = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        name  = props["name"]
        try:
            polygon  = shape(feature["geometry"])
            bbox     = polygon.bounds   # (minx, miny, maxx, maxy)
            zones[name] = {
                "geometry": polygon,
                "bbox":     bbox,
                "bbox_str": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "zone_id":  feature.get("id"),
                "center_id": props.get("center_id"),
            }
        except Exception as e:
            print(f"  Skipping zone {name}: {e}")

    print(f"Fetched {len(zones)} SNFAC zones")
    return zones


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all_winters(out_dir: Path, no_fetch: bool = False):  # no_fetch=True forces re-download -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch (or load cached) danger forecasts and observations for all winters.

    Returns
    -------
    (dangers_df, obs_df)
    """
    danger_cache = out_dir / "snfac_dangers_2021_2025.csv"
    obs_cache    = out_dir / "snfac_obs_2021_2025.csv"

    # ── Dangers ──────────────────────────────────────────────────────────────
    if danger_cache.exists() and not no_fetch:
        print(f"Loading cached dangers: {danger_cache}")
        dangers_df = pd.read_csv(danger_cache, parse_dates=["date"])
    else:
        print("Fetching danger forecasts for all winters...")
        season_dfs = []
        for start, end in tqdm(WINTERS, desc="Seasons"):
            df = get_dangers(CENTER_ID, start_date=start, end_date=end, verbose=False)
            season_dfs.append(df)
        dangers_df = pd.concat(season_dfs, ignore_index=True)
        dangers_df["date"] = pd.to_datetime(dangers_df["date"])
        dangers_df.to_csv(danger_cache, index=False)
        print(f"Saved dangers → {danger_cache}")

    # ── Observations ─────────────────────────────────────────────────────────
    # SNFAC_BBOX = "-115.5,43.5,-114.5,44.5"
    SNFAC_BBOX = "-115.4728,43.2953,-113.9101,44.527"


    if obs_cache.exists() and not no_fetch:
        print(f"Loading cached observations: {obs_cache}")
        obs_df = pd.read_csv(obs_cache, parse_dates=["date"])
    else:
        print("Fetching observations for all winters...")
        season_obs = []
        for start, end in tqdm(WINTERS, desc="Seasons"):
            df = get_observations(
                bbox=SNFAC_BBOX,
                start_date=start,
                end_date=end,
                verbose=False,
            )
            season_obs.append(df)
        obs_df = pd.concat(season_obs, ignore_index=True)
        obs_df["date"] = pd.to_datetime(obs_df["date"]).dt.normalize()
        obs_df.to_csv(obs_cache, index=False)
        print(f"Saved observations → {obs_cache}")

    return dangers_df, obs_df


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_high_danger_periods(
    dangers_df: pd.DataFrame,
    peak_height: float   = 3.0,
    peak_distance: int   = 14,
    peak_prominence: float = 0.3,
    rolling_window: int  = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify high-danger peak days from the danger forecast timeseries.

    Parameters
    ----------
    dangers_df      : DataFrame from get_dangers() with a 'danger_rating' column
    peak_height     : Minimum rolling-average danger rating to qualify as a peak
    peak_distance   : Minimum days between peaks
    peak_prominence : Minimum prominence of each peak
    rolling_window  : Days for rolling average

    Returns
    -------
    (daily_df, peak_dates_df)
      daily_df     — daily averaged danger with rolling mean
      peak_dates_df — rows for each detected peak, sorted by danger descending
    """
    df = dangers_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Max danger across above/near/below treeline elevations per row
    elev_cols = [c for c in df.columns if c in (
        "danger_above_current", "danger_near_current", "danger_below_current"
    )]
    df["danger_rating"] = df[elev_cols].max(axis=1)

    # Average max danger across zones per day
    daily = (
        df.groupby(df["date"].dt.date)["danger_rating"]
        .mean()
        .reset_index()
        .rename(columns={"date": "date", "danger_rating": "danger_mean"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_7d"] = daily["danger_mean"].rolling(
        rolling_window, center=True, min_periods=3
    ).mean()

    peaks, _ = find_peaks(
        daily["rolling_7d"].fillna(0),
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence,
    )

    peak_dates = daily.iloc[peaks][["date", "rolling_7d"]].copy()
    peak_dates = peak_dates.sort_values("rolling_7d", ascending=False).reset_index(drop=True)

    return daily, peak_dates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_danger_peaks(
    daily: pd.DataFrame,
    peak_dates: pd.DataFrame,
    daily_obs: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save a danger-peaks-vs-observations plot."""
    peaks_idx = daily[daily["date"].isin(peak_dates["date"])].index

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.fill_between(daily["date"], daily["rolling_7d"], alpha=0.2, color="steelblue")
    ax1.plot(daily["date"], daily["rolling_7d"], color="steelblue",
             linewidth=1.5, label="7-day rolling avg danger")
    ax1.scatter(daily.loc[peaks_idx, "date"], daily.loc[peaks_idx, "rolling_7d"],
                color="crimson", zorder=5, s=60, label="Peak periods")

    for level, label, color in [
        (2, "Moderate",     "#f5c518"),
        (3, "Considerable", "#ff8c00"),
        (4, "High",         "#cc0000"),
    ]:
        ax1.axhline(level, linestyle="--", linewidth=0.8, color=color, alpha=0.7)
        ax1.text(daily["date"].iloc[-1], level + 0.05, label,
                 color=color, fontsize=8, va="bottom")

    ax1.set_ylim(0, 5.2)
    ax1.set_ylabel("Danger Rating (avg across zones)")

    if not daily_obs.empty:
        ax2 = ax1.twinx()
        ax2.bar(daily_obs["date"], daily_obs["avalanche_count"],
                color="darkorange", alpha=0.5, width=1.5, label="Observed avalanches")
        ax2.set_ylabel("Observed Avalanche Count")
        ax2.set_ylim(0, max(daily_obs["avalanche_count"].max() * 1.5, 1))
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{CENTER_ID} — Danger Peaks vs Observed Avalanches (2021–2025)")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# sarvalanche runner
# ---------------------------------------------------------------------------

def run_sarvalanche(
    avalanche_date: str,
    bbox_str: str,
    zone_name: str,
    cache_dir: Path,
    overwrite: bool = False,
    static_fp = None,
) -> bool:
    """
    Run sarvalanche detection for a single zone and date.

    run_detection writes all outputs (*.nc, *.gpkg, opera/, arrays/,
    probabilities/) directly into cache_dir, using job_name as the file stem.

    Parameters
    ----------
    avalanche_date : ISO date string for the high-danger day (e.g. "2022-01-15")
    bbox_str       : Bounding box "minx,miny,maxx,maxy"
    zone_name      : Human-readable zone name — becomes part of the job_name /
                     output filename stem
    cache_dir      : Root cache directory passed to run_detection. All
                     intermediate and final files land here.
    overwrite      : Re-run even if {job_name}.nc already exists

    Returns
    -------
    True if the run completed (or was skipped), False on error
    """
    from sarvalanche.detection_pipeline import run_detection

    safe_name = zone_name.replace(" ", "_").replace("/", "-")
    job_name  = f"{safe_name}_{avalanche_date}"

    # run_detection uses cache_dir/{job_name}.nc as its output — skip if present
    expected_output = cache_dir / f"{job_name}.nc"
    if not overwrite and expected_output.exists() and expected_output.stat().st_size > 0:
        log.info(f"Skipping {job_name} — output already exists")
        return True

    # Parse bbox string → shapely box expected by run_detection
    try:
        minx, miny, maxx, maxy = [float(c) for c in bbox_str.split(",")]
        aoi = box(minx, miny, maxx, maxy)
    except ValueError as e:
        log.error(f"Invalid bbox '{bbox_str}' for {zone_name}: {e}")
        return False

    # try:
    ds = run_detection(
        aoi            = aoi,
        avalanche_date = avalanche_date,
        cache_dir      = cache_dir,
        job_name       = job_name,
        overwrite      = overwrite,
        static_fp      = static_fp
    )

    # ── Cleanup: release large in-memory objects before returning ────────────
    # flowpy and SAR stacking leave large arrays allocated; explicit cleanup
    # prevents OOM kills when running multiple zones back-to-back
    del ds
    gc.collect()
    return True
    # except Exception as e:
        # log.error(f"sarvalanche failed [{zone_name} {avalanche_date}]: {e}")
        # return False


def run_sarvalanche_for_peaks(
    peak_dates: pd.DataFrame,
    zones: dict,
    obs_df: pd.DataFrame,
    out_dir: Path,
    obs_window: int = 6,
) -> pd.DataFrame:
    """
    For each high-danger peak day × SNFAC zone:
      1. Fetch observations within ±obs_window days
      2. Run sarvalanche with the peak date as avalanche_date and zone bbox as AOI

    Parameters
    ----------
    peak_dates  : DataFrame with 'date' column (from find_high_danger_periods)
    zones       : Dict from fetch_snfac_zones()
                  {zone_name: {"bbox_str": ..., ...}}
    obs_df      : Full observations DataFrame (already loaded)
    out_dir     : Directory for sarvalanche outputs
    obs_window  : ±days around peak to include observations

    Returns
    -------
    DataFrame summarising every (peak_date, zone, obs_count) run
    """
    sarvalanche_dir = out_dir / "sarvalanche_runs"
    sarvalanche_dir.mkdir(parents=True, exist_ok=True)

    run_log: list[dict] = []

    for _, peak_row in tqdm(peak_dates.iterrows(), total=len(peak_dates), desc="Peak days"):
        peak_date = pd.Timestamp(peak_row["date"])
        date_str  = peak_date.date().isoformat()
        window_start = peak_date - timedelta(days=obs_window)
        window_end   = peak_date + timedelta(days=obs_window)

        # Observations within the window
        mask = (obs_df["date"] >= window_start) & (obs_df["date"] <= window_end)
        window_obs = obs_df[mask]


        for zone_name, zone_info in zones.items():
            safe_name = zone_name.replace(" ", "_").replace("/", "-")
            static_fp = next(sarvalanche_dir.glob(f"*{safe_name}*.nc"), None)
            obs_count = len(window_obs)   # could filter further by spatial intersection if needed

            # Run sarvalanche — cache_dir is the root for all job outputs
            run_sarvalanche(
                avalanche_date = date_str,
                bbox_str       = zone_info["bbox_str"],
                zone_name      = zone_name,
                cache_dir      = sarvalanche_dir,
                static_fp      = static_fp
            )

            run_log.append({
                "peak_date":       date_str,
                "danger_rolling":  peak_row["rolling_7d"],
                "zone_name":       zone_name,
                "zone_bbox":       zone_info["bbox_str"],
                "obs_window_start": window_start.date().isoformat(),
                "obs_window_end":   window_end.date().isoformat(),
                "obs_count":        obs_count,
            })

    run_log_df = pd.DataFrame(run_log)
    log_path   = out_dir / "sarvalanche_run_log.csv"
    run_log_df.to_csv(log_path, index=False)
    print(f"Run log saved → {log_path}  ({len(run_log_df)} total runs)")
    return run_log_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="high_danger_runs",
        description="Identify SNFAC high-danger periods and trigger sarvalanche runs",
    )
    parser.add_argument(
        "--center", default=CENTER_ID,
        help=f"Avalanche center ID (default: {CENTER_ID})"
    )
    parser.add_argument(
        "--obs-window", type=int, default=6, metavar="DAYS",
        help="±days of observations to gather around each peak (default: 6)"
    )
    parser.add_argument(
        "--peak-height", type=float, default=3.0,
        help="Min rolling-avg danger rating to qualify as a peak (default: 3.0)"
    )
    parser.add_argument(
        "--peak-distance", type=int, default=14,
        help="Min days between peaks (default: 14)"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=7,
        help="Rolling average window in days (default: 7)"
    )
    parser.add_argument(
        "--out-dir", default="./high_danger_output", metavar="DIR",
        help="Output directory (default: ./high_danger_output)"
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip API calls and load cached CSVs from --out-dir"
    )
    parser.add_argument(
        "--no-sarvalanche", action="store_true",
        help="Skip sarvalanche subprocess calls (useful for dry runs)"
    )
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch / load data ─────────────────────────────────────────────────
    dangers_df, obs_df = fetch_all_winters(out_dir, no_fetch=args.no_fetch)

    # Daily obs count for plotting
    daily_obs = (
        obs_df.groupby("date").size().reset_index(name="avalanche_count")
        if not obs_df.empty
        else pd.DataFrame(columns=["date", "avalanche_count"])
    )

    # ── 2. Detect high-danger peaks ──────────────────────────────────────────
    daily, peak_dates = find_high_danger_periods(
        dangers_df,
        peak_height    = args.peak_height,
        peak_distance  = args.peak_distance,
        rolling_window = args.rolling_window,
    )

    print(f"\nTop high-danger periods ({len(peak_dates)} peaks):")
    print(peak_dates.to_string(index=False))

    peak_dates.to_csv(out_dir / f"{CENTER_ID}_peak_dates.csv", index=False)

    # ── 3. Plot ──────────────────────────────────────────────────────────────
    plot_danger_peaks(
        daily, peak_dates, daily_obs,
        out_path=out_dir / f"{CENTER_ID}_danger_peaks_vs_obs.png",
    )

    # ── 4. Fetch zone geometries ─────────────────────────────────────────────
    zones = fetch_snfac_zones()

    # ── 5. sarvalanche runs ──────────────────────────────────────────────────
    if args.no_sarvalanche:
        print("\n--no-sarvalanche set: skipping subprocess calls.")
        print(f"Would run sarvalanche for {len(peak_dates)} peaks × {len(zones)} zones "
              f"= {len(peak_dates) * len(zones)} total runs")
    else:
        run_log = run_sarvalanche_for_peaks(
            peak_dates  = peak_dates,
            zones       = zones,
            obs_df      = obs_df,
            out_dir     = out_dir,
            obs_window  = args.obs_window,
        )
        print(run_log)


if __name__ == "__main__":
    main()