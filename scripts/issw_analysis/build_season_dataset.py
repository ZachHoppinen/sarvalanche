"""Build a season dataset (SAR + terrain + static probabilities) for a zone.

Assembles OPERA RTC-S1 SAR data, auxiliary terrain layers, runs FlowPy
runout modeling, TV despeckling, resolution weights, and static probabilities.
Outputs a ready-to-use season_dataset.nc and season_tracks.gpkg.

This is the "data assembly" step extracted from dual_tau_season_runner.py —
it does everything up to (but not including) per-step empirical computation
and ML inference.

Usage:
    conda run -n sarvalanche python scripts/issw_analysis/build_season_dataset.py \
        --zone "Sawtooth & Western Smoky Mtns" \
        --season 2024-2025 \
        --out-dir ./local/issw/dual_tau_output/

    # Reuse existing FlowPy tracks and static layers from a prior run
    conda run -n sarvalanche python scripts/issw_analysis/build_season_dataset.py \
        --zone "Sawtooth & Western Smoky Mtns" \
        --season 2024-2025 \
        --out-dir ./local/issw/dual_tau_output/ \
        --existing-runs-dir ./local/issw/high_danger_output/sarvalanche_runs/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("asf_search").setLevel(logging.WARNING)
logging.getLogger("rasterio.session").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.avalanche.org",
    "referer": "https://www.avalanche.org/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def fetch_center_zones(center_id: str = "SNFAC") -> dict:
    """Fetch forecast zone polygons from avalanche.org for any center.

    Parameters
    ----------
    center_id : str
        Avalanche center code (e.g. 'SNFAC', 'CNFAIC', 'GNFAC', 'CAIC').
    """
    url = f"https://api.avalanche.org/v2/public/products/map-layer/{center_id}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    geojson = r.json()

    zones = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        name = props["name"]
        try:
            polygon = shape(feature["geometry"])
            zones[name] = {"geometry": polygon, "bbox": polygon.bounds}
        except Exception as e:
            log.warning("Skipping zone %s: %s", name, e)

    log.info("Fetched %d %s zones", len(zones), center_id)
    return zones


# Backward-compatible alias
fetch_snfac_zones = fetch_center_zones


def season_nc_filename(season: str, zone: str) -> str:
    """Build canonical season dataset filename from season and zone name."""
    safe_zone = zone.replace(" ", "_").replace("/", "-").replace("&", "&")
    return f"season_{season}_{safe_zone}.nc"


def build_season_dataset(
    aoi,
    season_start: str,
    season_end: str,
    cache_dir: Path,
    crs: str = "EPSG:4326",
    resolution: float | None = None,
    static_fp: Path | None = None,
    track_gpkg: Path | None = None,
    baseline_days: int = 60,
    nc_filename: str = "season_dataset.nc",
    opera_cache_dir: Path | None = None,
) -> None:
    """Assemble full-season SAR + terrain, preprocess, compute static layers.

    Steps:
      1. Fetch SAR data (with baseline_days buffer on each end)
      2. Run FlowPy terrain modeling → season_tracks.gpkg
      3. TV despeckling
      4. Resolution weights
      5. Static probabilities (p_fcf, p_runout, p_slope)
      6. Drop tracks with < 3 acquisitions
      7. Save season_dataset.nc

    Parameters
    ----------
    opera_cache_dir : Path, optional
        Shared directory for OPERA tile downloads. When set, OPERA tiles
        are downloaded to ``opera_cache_dir/opera/`` instead of
        ``cache_dir/opera/``, enabling shared downloads across zones
        within the same center. Output nc is still written to cache_dir.
    """
    import geopandas as gpd

    from sarvalanche.io.dataset import assemble_dataset, load_netcdf_to_dataset
    from sarvalanche.io.export import export_netcdf
    from sarvalanche.io.load_data import cleanup_temp_files
    from sarvalanche.features.debris_flow_modeling import generate_runcount_alpha_angle
    from sarvalanche.preprocessing.pipelines import preprocess_rtc
    from sarvalanche.weights.local_resolution import get_local_resolution_weights
    from sarvalanche.probabilities.pipelines import get_static_probabilities
    from sarvalanche.utils.validation import validate_canonical

    season_nc = cache_dir / nc_filename

    fetch_start = (
        pd.Timestamp(season_start) - pd.Timedelta(days=baseline_days)
    ).strftime("%Y-%m-%d")
    fetch_end = min(
        pd.Timestamp(season_end) + pd.Timedelta(days=baseline_days),
        pd.Timestamp.now(),
    ).strftime("%Y-%m-%d")

    # ── 1. Assemble or load dataset ──────────────────────────────────────
    if season_nc.exists() and season_nc.stat().st_size > 0:
        log.info("Loading cached season dataset from %s", season_nc)
        ds = load_netcdf_to_dataset(season_nc)
    else:
        log.info(
            "Assembling season dataset: %s to %s (fetch from %s for baseline)",
            season_start, season_end, fetch_start,
        )
        if resolution is None:
            from pyproj import CRS as ProjCRS
            crs_obj = ProjCRS.from_user_input(crs)
            resolution = 30 if crs_obj.is_projected else 1 / 3600

        ds = assemble_dataset(
            aoi=aoi,
            start_date=fetch_start,
            stop_date=fetch_end,
            resolution=resolution,
            crs=crs,
            cache_dir=opera_cache_dir or cache_dir,
            static_layer_nc=static_fp,
            sar_only=False,
        )
        # Free memmap temp files immediately after assembly
        cleanup_temp_files()

    # Ensure time is datetime
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        log.warning("Converting time coordinate to datetime64")
        ds["time"] = pd.DatetimeIndex(ds["time"].values)

    # ── 2. FlowPy terrain modeling ───────────────────────────────────────
    _flowpy_vars = ["cell_counts", "runout_angle", "release_zones"]
    track_gpkg = track_gpkg or (cache_dir / "season_tracks.gpkg")
    missing_flowpy = not all(v in ds.data_vars for v in _flowpy_vars)
    gpkg_exists = track_gpkg.exists() and track_gpkg.stat().st_size > 0

    if not missing_flowpy and gpkg_exists:
        log.info("FlowPy vars present and gpkg exists — skipping")
    elif gpkg_exists and static_fp is not None and Path(static_fp).exists():
        log.info("Loading FlowPy variables from donor netcdf: %s", static_fp)
        donor_ds = load_netcdf_to_dataset(Path(static_fp))
        for v in _flowpy_vars:
            if v in donor_ds.data_vars:
                ds[v] = donor_ds[v]
                ds[v].attrs = donor_ds[v].attrs
        del donor_ds
        export_netcdf(ds, season_nc, overwrite=True)
    else:
        log.info("Running FlowPy terrain modeling")
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg, driver="GPKG")
        export_netcdf(ds, season_nc, overwrite=True)

    validate_canonical(ds)

    # ── 3. Preprocessing ─────────────────────────────────────────────────
    needs_save = False

    if ds.attrs.get("preprocessed") != "rtc_tv":
        log.info("Running TV despeckling")
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs["preprocessed"] = "rtc_tv"
        needs_save = True

    # ── 4. Resolution weights ────────────────────────────────────────────
    if "w_resolution" not in ds.data_vars:
        log.info("Computing resolution weights")
        ds["w_resolution"] = get_local_resolution_weights(ds["anf"])
        ds["w_resolution"].attrs = {
            "source": "sarvalanche",
            "units": "1",
            "product": "weight",
        }
        needs_save = True

    # ── 5. Static probabilities ──────────────────────────────────────────
    ref_date = pd.Timestamp(season_start) + (
        pd.Timestamp(season_end) - pd.Timestamp(season_start)
    ) / 2
    if "p_fcf" not in ds.data_vars:
        log.info("Computing static probabilities")
        ds = get_static_probabilities(ds, ref_date)
        needs_save = True

    if needs_save:
        log.info("Saving preprocessed dataset to %s", season_nc)
        export_netcdf(ds, season_nc, overwrite=True)

    # ── 6. Drop sparse tracks ────────────────────────────────────────────
    if "track" in ds.coords:
        all_times = pd.DatetimeIndex(ds["time"].values)
        track_arr = np.asarray(ds["track"].values)
        track_vals, track_counts = np.unique(track_arr, return_counts=True)
        log.info("Tracks: %s", dict(zip(track_vals, track_counts)))
        for tv in track_vals:
            track_times = all_times[track_arr == tv]
            log.info(
                "  Track %s: %d times, %s to %s",
                tv, len(track_times), track_times.min().date(), track_times.max().date(),
            )

        sparse_tracks = track_vals[track_counts < 3]
        if len(sparse_tracks) > 0:
            keep_mask = ~np.isin(track_arr, sparse_tracks)
            n_before = len(ds.time)
            ds = ds.sel(time=keep_mask)
            log.info(
                "Dropped %d sparse tracks (%s) with <3 acquisitions — %d→%d time steps",
                len(sparse_tracks), sparse_tracks.tolist(), n_before, len(ds.time),
            )

    # ── Summary ──────────────────────────────────────────────────────────
    all_times = pd.DatetimeIndex(ds["time"].values)
    season_start_ts = pd.Timestamp(season_start)
    n_baseline = int((all_times < season_start_ts).sum())
    log.info(
        "Season dataset ready: %d time steps (%d baseline + %d in-season), %d×%d spatial",
        len(ds.time), n_baseline, len(ds.time) - n_baseline,
        ds.sizes["y"], ds.sizes["x"],
    )
    log.info("Output: %s", season_nc)
    log.info("Tracks: %s", track_gpkg)


def main():
    parser = argparse.ArgumentParser(
        description="Build a season dataset (SAR + terrain + static probabilities)",
    )
    parser.add_argument(
        "--zone", required=True,
        help='Zone name (e.g. "Sawtooth & Western Smoky Mtns", "Turnagain Pass and Girdwood")',
    )
    parser.add_argument(
        "--center", default="SNFAC",
        help='Avalanche center ID (default: SNFAC). E.g. CNFAIC, GNFAC, CAIC.',
    )
    parser.add_argument(
        "--season", required=True,
        help='Winter season as "YYYY-YYYY" (e.g. "2024-2025")',
    )
    parser.add_argument(
        "--out-dir", default="./local/issw/dual_tau_output/",
        help="Output directory (default: ./local/issw/dual_tau_output/)",
    )
    parser.add_argument(
        "--existing-runs-dir", default=None,
        help="Path to existing sarvalanche_runs/ dir to reuse FlowPy .gpkg and static .nc",
    )
    parser.add_argument(
        "--crs", default="EPSG:4326",
        help="CRS for processing (default: EPSG:4326)",
    )
    parser.add_argument(
        "--baseline-days", type=int, default=60,
        help="Days before season start to fetch for SAR baseline (default: 60)",
    )
    args = parser.parse_args()

    # Parse season
    try:
        start_year, end_year = args.season.split("-")
        season_start = f"{start_year}-11-01"
        season_end = f"{end_year}-04-30"
    except ValueError:
        parser.error('--season must be "YYYY-YYYY" (e.g. "2024-2025")')

    # Fetch zone geometry
    zones = fetch_center_zones(args.center)
    if args.zone not in zones:
        available = "\n  ".join(sorted(zones.keys()))
        parser.error(f'Zone "{args.zone}" not found. Available:\n  {available}')

    aoi = zones[args.zone]["geometry"]
    safe_name = args.zone.replace(" ", "_").replace("/", "-")

    # Set up output directory
    out_dir = Path(args.out_dir)
    zone_cache = out_dir / safe_name
    zone_cache.mkdir(parents=True, exist_ok=True)

    # Find existing static files from prior runs
    static_fp = None
    track_gpkg = None
    _REQUIRED_STATIC = {"dem", "slope", "aspect", "fcf"}

    if args.existing_runs_dir:
        existing_dir = Path(args.existing_runs_dir)
        if existing_dir.is_dir():
            log.info("Searching %s for existing files", existing_dir)
            for match in sorted(existing_dir.glob(f"*{safe_name}*.gpkg")):
                track_gpkg = match
                log.info("Found existing track gpkg: %s", track_gpkg)
                break
            for match in sorted(existing_dir.glob(f"*{safe_name}*.nc")):
                try:
                    check_ds = xr.open_dataset(match)
                    has_static = _REQUIRED_STATIC.issubset(set(check_ds.data_vars))
                    check_ds.close()
                    if has_static:
                        static_fp = match
                        log.info("Found existing static nc: %s", static_fp)
                        break
                except Exception:
                    continue

    nc_fname = season_nc_filename(args.season, args.zone)

    build_season_dataset(
        aoi=aoi,
        season_start=season_start,
        season_end=season_end,
        cache_dir=zone_cache,
        crs=args.crs,
        static_fp=static_fp,
        track_gpkg=track_gpkg,
        baseline_days=args.baseline_days,
        nc_filename=nc_fname,
    )


if __name__ == "__main__":
    main()
