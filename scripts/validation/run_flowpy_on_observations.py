#!/usr/bin/env python3
"""Run FlowPy debris-flow modeling from each SNFAC observation point.

For every observation that falls within a netCDF's spatial and temporal extent,
this script:
  1. Clips the DEM to a buffer around the observation point.
  2. Creates a 5×5-pixel release zone centred on the observation.
  3. Runs FlowPy to generate debris-flow paths.
  4. Saves path GeoPackages grouped by day into local/issw/observations/.

Usage:
    conda run -n sarvalanche python scripts/validation/run_flowpy_on_observations.py
    conda run -n sarvalanche python scripts/validation/run_flowpy_on_observations.py --limit 10
"""

import argparse
import ast
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.vendored.flowpy import run_flowpy

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "local/issw/snfac_obs_2021_2025.csv"
NC_DIR = ROOT / "local/issw/netcdfs"
OUT_DIR = ROOT / "local/issw/observations"

NETCDF_FILES = sorted(NC_DIR.glob("*/season_*.nc"))

# Buffer around observation for DEM clip (pixels).  Must be large enough for
# FlowPy runout (~200 px ≈ 6 km at 30 m).
DEM_BUFFER_PX = 200
# Half-width of the 5×5 release zone (pixels from centre).
RELEASE_HALF = 2


def parse_location_point(loc_str: str) -> tuple[float, float]:
    """Return (lat, lng) from the CSV's location_point string."""
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def load_observations(csv_path: Path) -> gpd.GeoDataFrame:
    """Load and clean SNFAC observations into a GeoDataFrame."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["location_point", "date"])
    lats, lngs = [], []
    for loc in df["location_point"]:
        lat, lng = parse_location_point(loc)
        lats.append(lat)
        lngs.append(lng)
    df["lat"] = lats
    df["lng"] = lngs
    df["date"] = pd.to_datetime(df["date"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lng"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf


def obs_in_dataset(obs: gpd.GeoDataFrame, ds: xr.Dataset) -> gpd.GeoDataFrame:
    """Filter observations to those within the dataset's spatial + temporal extent."""
    dem = ds["dem"]
    left, bottom, right, top = dem.rio.bounds()
    time_min = pd.Timestamp(ds.time.values.min())
    time_max = pd.Timestamp(ds.time.values.max())

    mask = (
        (obs["lng"] >= left)
        & (obs["lng"] <= right)
        & (obs["lat"] >= bottom)
        & (obs["lat"] <= top)
        & (obs["date"] >= time_min)
        & (obs["date"] <= time_max)
    )
    return obs[mask].copy()


def clip_dem_around_point(
    dem: xr.DataArray, lng: float, lat: float, buffer_px: int
) -> xr.DataArray:
    """Clip the DEM to a box of ±buffer_px around (lng, lat)."""
    res_x, res_y = dem.rio.resolution()
    res_x, res_y = abs(res_x), abs(res_y)
    buf_x = buffer_px * res_x
    buf_y = buffer_px * res_y
    clipped = dem.rio.clip_box(
        minx=lng - buf_x,
        miny=lat - buf_y,
        maxx=lng + buf_x,
        maxy=lat + buf_y,
    )
    return clipped


def make_release_mask(
    dem_proj: xr.DataArray, obs_x: float, obs_y: float, half_px: int = RELEASE_HALF
) -> xr.DataArray:
    """Create a binary release mask with a (2*half_px+1)² block around (obs_x, obs_y).

    Parameters
    ----------
    dem_proj : projected DEM DataArray
    obs_x, obs_y : observation coordinates in the DEM's projected CRS
    half_px : half-width of the release zone in pixels
    """
    release = xr.zeros_like(dem_proj)
    x_vals = dem_proj.x.values
    y_vals = dem_proj.y.values
    col = int(np.argmin(np.abs(x_vals - obs_x)))
    row = int(np.argmin(np.abs(y_vals - obs_y)))
    r_lo = max(row - half_px, 0)
    r_hi = min(row + half_px + 1, len(y_vals))
    c_lo = max(col - half_px, 0)
    c_hi = min(col + half_px + 1, len(x_vals))
    release.values[r_lo:r_hi, c_lo:c_hi] = 1.0
    return release


def run_single_observation(
    dem: xr.DataArray,
    lng: float,
    lat: float,
    obs_id: str,
) -> gpd.GeoDataFrame | None:
    """Clip DEM, create release zone, run FlowPy for one observation."""
    # 1. Clip DEM around point (in original lon/lat coords)
    dem_clip = clip_dem_around_point(dem, lng, lat, DEM_BUFFER_PX)
    if dem_clip.size == 0:
        log.warning("Empty DEM clip for obs %s at (%.4f, %.4f)", obs_id, lng, lat)
        return None

    # 2. Reproject to UTM
    utm_crs = dem_clip.rio.estimate_utm_crs()
    dem_proj = dem_clip.rio.reproject(utm_crs)

    # 3. Transform observation point to projected CRS
    obs_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lng], [lat]), crs="EPSG:4326"
    ).to_crs(utm_crs)
    obs_x = obs_gdf.geometry.iloc[0].x
    obs_y = obs_gdf.geometry.iloc[0].y

    # 4. Build 5×5 release mask
    release = make_release_mask(dem_proj, obs_x, obs_y, half_px=RELEASE_HALF)
    n_release = int((release > 0).sum())
    if n_release == 0:
        log.warning("No release pixels for obs %s", obs_id)
        return None

    log.info(
        "Running FlowPy for obs %s: DEM shape=%s, release pixels=%d",
        obs_id, dem_proj.shape, n_release,
    )

    # 5. Run FlowPy
    cell_counts, fp_ta, path_list = run_flowpy(
        dem=dem_proj, release=release, alpha=20, max_workers=4
    )

    # 6. Merge path GeoDataFrames
    valid_paths = [p for p in path_list if p is not None]
    if not valid_paths:
        log.warning("FlowPy returned no paths for obs %s", obs_id)
        return None

    paths_gdf = gpd.GeoDataFrame(
        pd.concat(valid_paths, ignore_index=True), crs=valid_paths[0].crs
    )
    paths_gdf["obs_id"] = obs_id
    return paths_gdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N observations total (for testing).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obs = load_observations(CSV_PATH)
    log.info("Loaded %d observations with valid location + date", len(obs))

    all_results = []
    processed = 0

    for nc_path in NETCDF_FILES:
        if args.limit is not None and processed >= args.limit:
            break
        if not nc_path.exists():
            log.warning("NetCDF not found: %s", nc_path)
            continue

        log.info("Opening %s", nc_path.name)
        ds = xr.open_dataset(nc_path)
        dem = ds["dem"].rio.write_crs("EPSG:4326")

        matched = obs_in_dataset(obs, ds)
        log.info("  %d observations matched spatially + temporally", len(matched))

        for _, row in matched.iterrows():
            if args.limit is not None and processed >= args.limit:
                break
            obs_id = row["id"]
            processed += 1
            log.info("=== Observation %d/%s: %s ===", processed, args.limit or "all", obs_id)
            try:
                paths_gdf = run_single_observation(
                    dem=dem,
                    lng=row["lng"],
                    lat=row["lat"],
                    obs_id=obs_id,
                )
                if paths_gdf is not None:
                    paths_gdf["date"] = row["date"].strftime("%Y-%m-%d")
                    paths_gdf["location_name"] = row.get("location_name", "")
                    paths_gdf["zone_name"] = row.get("zone_name", "")
                    all_results.append(paths_gdf)
            except Exception:
                log.exception("Failed for obs %s", obs_id)

        ds.close()

    if not all_results:
        log.warning("No results produced.")
        return

    combined = gpd.GeoDataFrame(
        pd.concat(all_results, ignore_index=True),
        crs=all_results[0].crs,
    )
    log.info("Total paths: %d from %d observations", len(combined), combined["obs_id"].nunique())

    # Group by day and save
    for date_str, group in combined.groupby("date"):
        out_path = OUT_DIR / f"{date_str}_flowpy_paths.gpkg"
        group.to_file(out_path, driver="GPKG")
        log.info("Saved %d paths for %s -> %s", len(group), date_str, out_path)

    # Also save a combined file
    combined_path = OUT_DIR / "all_flowpy_paths.gpkg"
    combined.to_file(combined_path, driver="GPKG")
    log.info("Saved combined file -> %s", combined_path)


if __name__ == "__main__":
    main()
