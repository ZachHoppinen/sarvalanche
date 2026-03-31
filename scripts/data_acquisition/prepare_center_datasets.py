"""
prepare_center_datasets.py — Fetch observations, dangers, and zone geometries
for any avalanche center, then call prepare_dataset for each zone × winter to
build full-season preprocessed datasets for ML training.

CLI usage:
    python prepare_center_datasets.py --center UAC --out-dir local/uac/datasets
    python prepare_center_datasets.py --center UAC --no-fetch     # use cached CSVs
    python prepare_center_datasets.py --center UAC --no-prepare   # fetch only, skip prepare_dataset
    python prepare_center_datasets.py --center SNFAC              # works for any center
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm.auto import tqdm

# Sibling script imports (same pattern as high/low_danger_runs.py)
sys.path.append(str(Path(__file__).resolve().parent))
from get_avalanche_observations import get_observations
from get_forecast_dangers import get_dangers
from get_forecast_zone_geojson import fetch_avalanche_zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CENTER = "UAC"

# Winter seasons: Nov 1 → May 1
WINTERS = [
    ("2023-11-01", "2024-05-01"),
    ("2024-11-01", "2025-05-01"),
]

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.avalanche.org",
    "referer": "https://www.avalanche.org/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


# ---------------------------------------------------------------------------
# Zone fetching
# ---------------------------------------------------------------------------

def fetch_zones(center_id: str) -> dict:
    """
    Fetch forecast zone polygons for any avalanche center.

    Returns
    -------
    dict : {zone_name: {"geometry": polygon, "bbox": tuple,
            "bbox_str": str, "zone_id": ..., "center_id": ...}}
    """
    raw_zones = fetch_avalanche_zones(center_id=center_id)

    zones = {}
    for name, info in raw_zones.items():
        polygon = info["geometry"]
        bbox = polygon.bounds
        zones[name] = {
            "geometry": polygon,
            "bbox": bbox,
            "bbox_str": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "shapely_box": box(*bbox),
            "zone_id": info.get("center_id"),
            "center_id": center_id,
        }

    print(f"Fetched {len(zones)} zones for {center_id}:")
    for name, info in zones.items():
        print(f"  {name:40s}  bbox: {[round(c, 3) for c in info['bbox']]}")
    return zones


# ---------------------------------------------------------------------------
# Data fetching (dangers + observations)
# ---------------------------------------------------------------------------

def fetch_data(
    center_id: str,
    zones: dict,
    winters: list[tuple[str, str]],
    out_dir: Path,
    no_fetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch or load cached danger forecasts and observations for all winters.

    Parameters
    ----------
    center_id : Avalanche center ID (e.g. "UAC", "SNFAC")
    zones     : Dict from fetch_zones()
    winters   : List of (start_date, end_date) tuples
    out_dir   : Directory for cached CSVs
    no_fetch  : If True, only load existing caches (error if missing)

    Returns
    -------
    (dangers_df, obs_df)
    """
    cid = center_id.lower()
    danger_cache = out_dir / f"{cid}_dangers.csv"
    obs_cache = out_dir / f"{cid}_observations.csv"

    # ── Dangers ──────────────────────────────────────────────────────────────
    if danger_cache.exists() and danger_cache.stat().st_size > 1 and not no_fetch:
        print(f"Loading cached dangers: {danger_cache}")
        dangers_df = pd.read_csv(danger_cache, parse_dates=["date"])
    else:
        print(f"Fetching danger forecasts for {center_id}...")
        season_dfs = []
        for start, end in tqdm(winters, desc="Seasons"):
            df = get_dangers(center_id, start_date=start, end_date=end, verbose=False)
            season_dfs.append(df)
        dangers_df = pd.concat(season_dfs, ignore_index=True)
        if not dangers_df.empty and "date" in dangers_df.columns:
            dangers_df["date"] = pd.to_datetime(dangers_df["date"])
        dangers_df.to_csv(danger_cache, index=False)
        print(f"Saved dangers → {danger_cache}  ({len(dangers_df)} rows)")

    # ── Observations ─────────────────────────────────────────────────────────
    # Derive bbox from the union of all zone geometries
    combined = unary_union([z["geometry"] for z in zones.values()])
    obs_bbox = combined.bounds  # (minx, miny, maxx, maxy)
    obs_bbox_str = f"{obs_bbox[0]},{obs_bbox[1]},{obs_bbox[2]},{obs_bbox[3]}"
    print(f"Observation bbox (union of all zones): {obs_bbox_str}")

    if obs_cache.exists() and obs_cache.stat().st_size > 1 and not no_fetch:
        print(f"Loading cached observations: {obs_cache}")
        obs_df = pd.read_csv(obs_cache, parse_dates=["date"])
    else:
        print(f"Fetching observations for {center_id}...")
        season_obs = []
        for start, end in tqdm(winters, desc="Seasons"):
            df = get_observations(
                bbox=obs_bbox_str,
                start_date=start,
                end_date=end,
                verbose=False,
            )
            season_obs.append(df)
        obs_df = pd.concat(season_obs, ignore_index=True)
        if not obs_df.empty:
            obs_df["date"] = pd.to_datetime(obs_df["date"]).dt.normalize()
        obs_df.to_csv(obs_cache, index=False)
        print(f"Saved observations → {obs_cache}  ({len(obs_df)} rows)")

    return dangers_df, obs_df


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_all_datasets(
    zones: dict,
    winters: list[tuple[str, str]],
    out_dir: Path,
    overwrite: bool = False,
    opera_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    For each zone × winter, call build_season_dataset to build a preprocessed
    full-season xr.Dataset with SAR, terrain, FlowPy, TV despeckle,
    w_resolution, and static probabilities.

    Parameters
    ----------
    zones    : Dict from fetch_zones()
    winters  : List of (start_date, end_date) tuples
    out_dir  : Root output directory
    overwrite : Force recomputation

    Returns
    -------
    DataFrame run log with one row per (zone, winter)
    """
    # Import build_season_dataset from sibling script
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent / "issw_analysis")
    )
    from build_season_dataset import build_season_dataset, season_nc_filename

    datasets_dir = out_dir / "netcdfs"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    run_log: list[dict] = []

    total = len(zones) * len(winters)
    pbar = tqdm(total=total, desc="Preparing datasets", unit="run")

    for zone_name, zone_info in zones.items():
        safe_name = zone_name.replace(" ", "_").replace("/", "-")
        zone_dir = datasets_dir / safe_name
        zone_dir.mkdir(parents=True, exist_ok=True)

        for start_date, end_date in winters:
            winter_label = f"{start_date[:4]}-{end_date[:4]}"
            nc_fname = season_nc_filename(winter_label, zone_name)

            # Check if output already exists
            expected_nc = zone_dir / nc_fname
            if not overwrite and expected_nc.exists() and expected_nc.stat().st_size > 0:
                log.info("Skipping %s — output already exists", nc_fname)
                run_log.append({
                    "zone_name": zone_name,
                    "winter": winter_label,
                    "status": "skipped",
                    "nc_path": str(expected_nc),
                })
                pbar.update(1)
                continue

            # Look for existing static_fp and track_gpkg from prior runs
            static_fp = next(zone_dir.glob(f"*.nc"), None)
            track_gpkg = next(zone_dir.glob(f"*.gpkg"), None)

            try:
                build_season_dataset(
                    aoi=zone_info["geometry"],
                    season_start=start_date,
                    season_end=end_date,
                    cache_dir=zone_dir,
                    static_fp=static_fp,
                    track_gpkg=track_gpkg,
                    nc_filename=nc_fname,
                    opera_cache_dir=opera_cache_dir,
                )
                status = "ok"
                log.info("Completed %s", nc_fname)
            except Exception as e:
                log.error("Failed %s: %s", nc_fname, e)
                status = f"error: {e}"

            gc.collect()

            run_log.append({
                "zone_name": zone_name,
                "winter": winter_label,
                "status": status,
                "nc_path": str(expected_nc),
            })
            pbar.update(1)

    pbar.close()

    run_log_df = pd.DataFrame(run_log)
    log_path = out_dir / "prepare_run_log.csv"
    run_log_df.to_csv(log_path, index=False)
    print(f"\nRun log saved → {log_path}  ({len(run_log_df)} total runs)")
    return run_log_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_center_datasets",
        description="Fetch avalanche data and prepare full-season datasets for any center",
    )
    parser.add_argument(
        "--center", default=DEFAULT_CENTER,
        help=f"Avalanche center ID (default: {DEFAULT_CENTER})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: local/issw/{center})",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip API calls, load cached CSVs",
    )
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="Only fetch data, skip prepare_dataset calls",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Force recomputation of existing datasets",
    )
    parser.add_argument(
        "--exclude-zones", nargs="*", default=[],
        metavar="ZONE",
        help="Zone names to exclude (e.g. --exclude-zones Southwest Skyline)",
    )
    parser.add_argument(
        "--include-zones", nargs="*", default=None,
        metavar="ZONE",
        help='Zone names to include (only these zones will be processed)',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    center_id = args.center.upper()
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"local/issw/{center_id.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch zone geometries ─────────────────────────────────────────────
    zones = fetch_zones(center_id)

    # Apply zone filters
    if args.include_zones:
        included = set(args.include_zones)
        zones = {k: v for k, v in zones.items() if k in included}
        missing = included - set(zones.keys())
        if missing:
            print(f"WARNING: zones not found: {missing}")
        print(f"Including {len(zones)} zone(s): {list(zones.keys())}")
    if args.exclude_zones:
        excluded = set(args.exclude_zones)
        before = len(zones)
        zones = {k: v for k, v in zones.items() if k not in excluded}
        print(f"Excluded {before - len(zones)} zone(s): {excluded}")
        print(f"Remaining: {list(zones.keys())}")

    # ── 2. Fetch dangers + observations ──────────────────────────────────────
    dangers_df, obs_df = fetch_data(
        center_id=center_id,
        zones=zones,
        winters=WINTERS,
        out_dir=out_dir,
        no_fetch=args.no_fetch,
    )

    print(f"\nDangers: {len(dangers_df)} rows")
    print(f"Observations: {len(obs_df)} rows")

    # ── 3. Prepare datasets ──────────────────────────────────────────────────
    if args.no_prepare:
        print("\n--no-prepare set: skipping prepare_dataset calls.")
        print(f"Would prepare {len(zones)} zones × {len(WINTERS)} winters "
              f"= {len(zones) * len(WINTERS)} datasets")
    else:
        # Shared OPERA cache at center level to avoid duplicate downloads
        opera_cache = out_dir / "netcdfs"
        opera_cache.mkdir(parents=True, exist_ok=True)
        run_log = prepare_all_datasets(
            zones=zones,
            winters=WINTERS,
            out_dir=out_dir,
            overwrite=args.overwrite,
            opera_cache_dir=opera_cache,
        )
        print(run_log)


if __name__ == "__main__":
    main()
