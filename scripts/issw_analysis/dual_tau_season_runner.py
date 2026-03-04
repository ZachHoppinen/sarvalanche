"""
dual_tau_season_runner.py — Run dual-tau (slow + fast) season analysis for
avalanche detection using SAR backscatter change and ML inference.

Instead of comparing high vs low danger periods, this script runs two parallel
temporal weighting regimes (tau_slow=24d, tau_fast=3d) at every SAR acquisition
step through the season. The tau=24 result serves as a prior for the tau=3
inference. Outputs a per-polygon season inventory and rolling time series.

CLI usage:
    conda run -n sarvalanche python scripts/issw_analysis/dual_tau_season_runner.py \
        --zone "Sawtooth" \
        --season "2024-2025" \
        --tau-slow 24 --tau-fast 3 \
        --out-dir ./local/issw/dual_tau_output/ \
        --existing-runs-dir ./local/issw/high_danger_output/sarvalanche_runs/
"""

import argparse
import gc
import logging
import pickle
import re

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import shape
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("asf_search").setLevel(logging.WARNING)
logging.getLogger("rasterio.session").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENTER_ID = "SNFAC"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.avalanche.org",
    "referer": "https://www.avalanche.org/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Variables that depend on temporal weights and must be cleaned between steps
_STALE_PATTERNS = [
    re.compile(r"^p_\d+_V[VH]_empirical$"),
    re.compile(r"^d_\d+_V[VH]_empirical$"),
]
_STALE_EXACT = {
    "p_empirical",
    "d_empirical",
    "unmasked_p_target",
    "p_pixelwise",
    "detections",
    "w_temporal",
}


# ---------------------------------------------------------------------------
# Zone geometry helpers
# ---------------------------------------------------------------------------


def fetch_snfac_zones() -> dict:
    """Fetch SNFAC forecast zone polygons from avalanche.org."""
    url = f"https://api.avalanche.org/v2/public/products/map-layer/{CENTER_ID}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    geojson = r.json()

    zones = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        name = props["name"]
        try:
            polygon = shape(feature["geometry"])
            bbox = polygon.bounds
            zones[name] = {
                "geometry": polygon,
                "bbox": bbox,
                "bbox_str": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "zone_id": feature.get("id"),
                "center_id": props.get("center_id"),
            }
        except Exception as e:
            print(f"  Skipping zone {name}: {e}")

    print(f"Fetched {len(zones)} SNFAC zones")
    return zones


# ---------------------------------------------------------------------------
# Dataset preparation (compute-once)
# ---------------------------------------------------------------------------


def _clean_stale_variables(ds: xr.Dataset) -> xr.Dataset:
    """Remove variables that depend on temporal weights."""
    to_drop = []
    for var in list(ds.data_vars):
        if var in _STALE_EXACT:
            to_drop.append(var)
            continue
        for pat in _STALE_PATTERNS:
            if pat.match(var):
                to_drop.append(var)
                break
    if to_drop:
        log.debug("Dropping %d stale variables: %s", len(to_drop), to_drop)
        ds = ds.drop_vars(to_drop)
    return ds


def prepare_season_dataset(
    aoi,
    season_start: str,
    season_end: str,
    cache_dir: Path,
    crs: str = "EPSG:4326",
    resolution: float | None = None,
    static_fp: Path | None = None,
    track_gpkg: Path | None = None,
    baseline_days: int = 60,
) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """
    Assemble full-season SAR data + terrain, run FlowPy, preprocess, compute
    static weights and probabilities. All compute-once work.

    Fetches SAR data starting ``baseline_days`` before ``season_start`` so the
    SAR transformer has enough baseline pairs for the earliest season steps.

    Returns (ds, paths_gdf).
    """
    from sarvalanche.io.dataset import assemble_dataset, load_netcdf_to_dataset
    from sarvalanche.io.export import export_netcdf
    from sarvalanche.features.debris_flow_modeling import generate_runcount_alpha_angle
    from sarvalanche.preprocessing.pipelines import preprocess_rtc
    from sarvalanche.weights.local_resolution import get_local_resolution_weights
    from sarvalanche.probabilities.pipelines import get_static_probabilities
    from sarvalanche.utils.validation import validate_canonical

    # Determine output paths
    season_nc = cache_dir / "season_dataset.nc"

    # Fetch window: start baseline_days before season to give the SAR
    # transformer enough pre-event acquisitions for early season steps.
    fetch_start = (
        pd.Timestamp(season_start) - pd.Timedelta(days=baseline_days)
    ).strftime("%Y-%m-%d")
    fetch_end = (
        pd.Timestamp(season_end) + pd.Timedelta(days=baseline_days)
    ).strftime("%Y-%m-%d")

    # ── 1. Assemble or load dataset ───────────────────────────────────────
    if season_nc.exists() and season_nc.stat().st_size > 0:
        log.info("Loading cached season dataset from %s", season_nc)
        ds = load_netcdf_to_dataset(season_nc)
    else:
        log.info(
            "Assembling season dataset: %s to %s (fetch from %s for baseline)",
            season_start,
            season_end,
            fetch_start,
        )

        if resolution is None:
            from pyproj import CRS

            crs_obj = CRS.from_user_input(crs)
            resolution = 30 if crs_obj.is_projected else 1 / 3600

        ds = assemble_dataset(
            aoi=aoi,
            start_date=fetch_start,
            stop_date=fetch_end,
            resolution=resolution,
            crs=crs,
            cache_dir=cache_dir,
            static_layer_nc=static_fp,
            sar_only=False,
        )

    # Ensure time coordinate is datetime (survives netCDF roundtrip)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        log.warning(
            "time coordinate has dtype %s — converting to datetime64",
            ds["time"].dtype,
        )
        ds["time"] = pd.DatetimeIndex(ds["time"].values)

    # ── 2. FlowPy terrain polygons ────────────────────────────────────────
    _flowpy_vars = ["cell_counts", "runout_angle", "release_zones"]
    track_gpkg = track_gpkg or (cache_dir / "season_tracks.gpkg")
    missing_flowpy = not all(v in ds.data_vars for v in _flowpy_vars)
    gpkg_exists = track_gpkg.exists() and track_gpkg.stat().st_size > 0

    if not missing_flowpy and gpkg_exists:
        log.info("FlowPy vars present and gpkg exists — loading paths_gdf")
        paths_gdf = gpd.read_file(track_gpkg)
    elif gpkg_exists and static_fp is not None and Path(static_fp).exists():
        log.info("Loading FlowPy variables from donor netcdf: %s", static_fp)
        donor_ds = load_netcdf_to_dataset(Path(static_fp))
        for v in _flowpy_vars:
            if v in donor_ds.data_vars:
                ds[v] = donor_ds[v]
                ds[v].attrs = donor_ds[v].attrs
        del donor_ds
        paths_gdf = gpd.read_file(track_gpkg)
        log.info("Saving dataset with FlowPy vars to %s", season_nc)
        export_netcdf(ds, season_nc, overwrite=True)
    else:
        log.info("Running FlowPy terrain modeling")
        ds, paths_gdf = generate_runcount_alpha_angle(ds)
        paths_gdf.to_file(track_gpkg, driver="GPKG")
        log.info("Saving dataset with FlowPy vars to %s", season_nc)
        export_netcdf(ds, season_nc, overwrite=True)

    validate_canonical(ds)

    # ── 3. Preprocessing ──────────────────────────────────────────────────
    needs_save = False

    if ds.attrs.get("preprocessed") != "rtc_tv":
        log.info("Running TV despeckling")
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs["preprocessed"] = "rtc_tv"
        needs_save = True

    # ── 4. Static resolution weights ──────────────────────────────────────
    if "w_resolution" not in ds.data_vars:
        log.info("Computing resolution weights")
        ds["w_resolution"] = get_local_resolution_weights(ds["anf"])
        ds["w_resolution"].attrs = {
            "source": "sarvalanche",
            "units": "1",
            "product": "weight",
        }
        needs_save = True

    # ── 5. Static probabilities ───────────────────────────────────────────
    # Use mid-season date as reference (doesn't affect terrain-only probs)
    ref_date = pd.Timestamp(season_start) + (
        pd.Timestamp(season_end) - pd.Timestamp(season_start)
    ) / 2
    if "p_fcf" not in ds.data_vars:
        log.info("Computing static probabilities")
        ds = get_static_probabilities(ds, ref_date)
        needs_save = True

    # Persist so subsequent runs skip preprocessing/weights/static probs
    if needs_save:
        log.info("Saving preprocessed dataset to %s", season_nc)
        export_netcdf(ds, season_nc, overwrite=True)

    # ── 6. Drop tracks with < 3 acquisitions ────────────────────────────
    # Tracks with too few time steps can never form crossing pairs and
    # cause ValueError in backscatter_changes_crossing_date.
    if "track" in ds.coords:
        track_vals, track_counts = np.unique(
            ds["track"].values, return_counts=True
        )
        sparse_tracks = track_vals[track_counts < 3]
        if len(sparse_tracks) > 0:
            keep_mask = ~ds["track"].isin(sparse_tracks)
            n_before = len(ds.time)
            ds = ds.sel(time=keep_mask)
            log.info(
                "Dropped %d sparse tracks (%s) with <3 acquisitions — "
                "%d→%d time steps",
                len(sparse_tracks),
                sparse_tracks.tolist(),
                n_before,
                len(ds.time),
            )

    # Load into memory — the dataset is ~2 GB but feature extraction
    # needs random pixel access per track; dask lazy loading is 10-100x slower.
    if any(var.chunks is not None for var in ds.variables.values()):
        log.info("Loading dataset into memory")
        ds = ds.load()

    all_times = pd.DatetimeIndex(ds["time"].values)
    n_baseline = int((all_times < pd.Timestamp(season_start)).sum())
    log.info(
        "Season dataset ready: %d time steps (%d baseline + %d in-season), %d×%d spatial",
        len(ds.time),
        n_baseline,
        len(ds.time) - n_baseline,
        ds.sizes["y"],
        ds.sizes["x"],
    )
    return ds, paths_gdf


# ---------------------------------------------------------------------------
# Per-step SAR feature computation
# ---------------------------------------------------------------------------


def compute_empirical_features(
    ds: xr.Dataset,
    reference_date: np.datetime64,
    tau_days: float,
) -> xr.Dataset | None:
    """
    Compute tau-dependent SAR features for a single step/tau combination.

    Cleans stale empirical intermediates, computes temporal weights and
    empirical backscatter probability.
    Returns modified ds, or None on failure.
    """
    from sarvalanche.weights.temporal import get_temporal_weights
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )

    # Clean tau-dependent variables (keep ML distances)
    _empirical_stale = {
        "p_empirical", "d_empirical", "unmasked_p_target",
        "p_pixelwise", "detections", "w_temporal",
    }
    to_drop = [v for v in ds.data_vars if v in _empirical_stale]
    for pat in _STALE_PATTERNS:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        log.debug("Dropping %d stale empirical variables", len(to_drop))
        ds = ds.drop_vars(to_drop)

    # Temporal weights for this step/tau
    ds["w_temporal"] = get_temporal_weights(
        ds["time"], reference_date, tau_days=tau_days
    )
    ds["w_temporal"].attrs = {
        "source": "sarvalanche",
        "units": "1",
        "product": "weight",
    }

    # Empirical backscatter probability
    log.info(
        "Computing empirical backscatter (date=%s, tau=%.0f)",
        reference_date,
        tau_days,
    )
    try:
        ds["p_empirical"], ds["d_empirical"] = (
            calculate_empirical_backscatter_probability(
                ds,
                reference_date,
                use_agreement_boosting=True,
                agreement_strength=0.8,
                min_prob_threshold=0.2,
            )
        )
    except ValueError as e:
        log.warning(
            "Skipping step date=%s tau=%.0f — empirical backscatter failed: %s",
            reference_date,
            tau_days,
            e,
        )
        return None

    return ds


# ---------------------------------------------------------------------------
# Accumulation helpers
# ---------------------------------------------------------------------------


def accumulate_inventory(
    inventory: dict,
    result_gdf: gpd.GeoDataFrame,
    step_date: pd.Timestamp,
    tau_label: str,
    threshold: float = 0.5,
) -> None:
    """
    Update per-track running stats dict with this step's results.

    inventory[track_idx] = {
        'slow_confidences': [...], 'fast_confidences': [...],
        'slow_dates': [...], 'fast_dates': [...],
        ...
    }
    """
    for idx, row in result_gdf.iterrows():
        if idx not in inventory:
            inventory[idx] = {
                "slow_confidences": [],
                "fast_confidences": [],
                "slow_dates": [],
                "fast_dates": [],
            }
        rec = inventory[idx]
        p = float(row["p_debris"])

        if tau_label == "slow":
            rec["slow_confidences"].append(p)
            rec["slow_dates"].append(step_date)
        else:
            rec["fast_confidences"].append(p)
            rec["fast_dates"].append(step_date)


def build_inventory_gdf(
    inventory: dict,
    paths_gdf: gpd.GeoDataFrame,
    threshold: float = 0.5,
) -> gpd.GeoDataFrame:
    """Convert accumulated dict → final inventory GeoDataFrame."""
    rows = []
    for idx, rec in inventory.items():
        if idx not in paths_gdf.index:
            continue

        slow = np.array(rec["slow_confidences"])
        fast = np.array(rec["fast_confidences"])
        fast_dates = rec["fast_dates"]

        row = {
            "track_idx": idx,
            "ran_this_season": bool(np.any(fast >= threshold)) if len(fast) > 0 else False,
            "slow_confidence_max": float(slow.max()) if len(slow) > 0 else 0.0,
            "slow_confidence_mean": float(slow.mean()) if len(slow) > 0 else 0.0,
            "fast_confidence_max": float(fast.max()) if len(fast) > 0 else 0.0,
            "fast_confidence_mean": float(fast.mean()) if len(fast) > 0 else 0.0,
            "first_detection_date": (
                min(d for d, p in zip(fast_dates, fast) if p >= threshold)
                if len(fast) > 0 and np.any(fast >= threshold)
                else pd.NaT
            ),
            "peak_detection_date": (
                fast_dates[int(fast.argmax())] if len(fast) > 0 else pd.NaT
            ),
            "n_active_slow": int((slow >= threshold).sum()) if len(slow) > 0 else 0,
            "n_active_fast": int((fast >= threshold).sum()) if len(fast) > 0 else 0,
        }
        rows.append(row)

    inv_df = pd.DataFrame(rows)
    # Merge with paths_gdf geometry
    inv_gdf = paths_gdf[["geometry"]].copy()
    inv_gdf["track_idx"] = inv_gdf.index
    if len(inv_df) > 0:
        inv_gdf = inv_gdf.merge(inv_df, on="track_idx", how="left")
    inv_gdf = gpd.GeoDataFrame(inv_gdf, geometry="geometry", crs=paths_gdf.crs)
    return inv_gdf


def build_timeseries_df(records: list[dict]) -> pd.DataFrame:
    """Convert list of per-step dicts → DataFrame."""
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_dual_tau_season(
    aoi,
    zone_name: str,
    season_start: str,
    season_end: str,
    cache_dir: Path,
    tau_slow: float = 24,
    tau_fast: float = 3,
    crs: str = "EPSG:4326",
    resolution: float | None = None,
    static_fp: Path | None = None,
    track_gpkg: Path | None = None,
    detection_threshold: float = 0.9,
    timeseries_threshold: float = 0.9,
    dry_run: bool = False,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Run dual-tau season analysis for one zone.

    For each SAR acquisition step:
      1. Compute SAR features with tau_slow → ML inference (slow result)
      2. Compute SAR features with tau_fast, using slow result as prior → ML inference (fast result)

    Returns (inventory_gdf, timeseries_df).
    """
    safe_name = zone_name.replace(" ", "_").replace("/", "-")

    # ── 1. Prepare dataset ────────────────────────────────────────────────
    log.info("Preparing season dataset for %s", zone_name)
    ds, paths_gdf = prepare_season_dataset(
        aoi=aoi,
        season_start=season_start,
        season_end=season_end,
        cache_dir=cache_dir,
        crs=crs,
        resolution=resolution,
        static_fp=static_fp,
        track_gpkg=track_gpkg,
    )

    # ── 2. Load XGBoost classifier ────────────────────────────────────────
    import joblib
    from sarvalanche.ml.track_classifier import predict_tracks, TRACK_PREDICTOR_MODEL
    log.info("Loading XGBoost classifier from %s", TRACK_PREDICTOR_MODEL)
    xgb_clf = joblib.load(TRACK_PREDICTOR_MODEL)

    # ── 3. Derive step dates ──────────────────────────────────────────────
    season_start_ts = pd.Timestamp(season_start)
    season_end_ts = pd.Timestamp(season_end)
    all_times = pd.DatetimeIndex(ds["time"].values)
    step_dates = all_times[
        (all_times >= season_start_ts) & (all_times <= season_end_ts)
    ]

    log.info(
        "Season window: %s to %s — %d SAR acquisition steps",
        season_start,
        season_end,
        len(step_dates),
    )

    if dry_run:
        print(f"\n[DRY RUN] Zone: {zone_name}")
        print(f"  Season: {season_start} to {season_end}")
        print(f"  Total SAR times in dataset: {len(all_times)}")
        print(f"  Steps in season window: {len(step_dates)}")
        print(f"  tau_slow={tau_slow}, tau_fast={tau_fast}")
        print(f"  Tracks: {len(paths_gdf)}")
        print("\n  Step dates:")
        for d in step_dates:
            print(f"    {d.date()}")
        return gpd.GeoDataFrame(), pd.DataFrame()

    # ── 4. Step loop (with checkpointing) ────────────────────────────────
    checkpoint_path = cache_dir / "checkpoint.pkl"
    completed_dates: set = set()
    inventory: dict = {}
    timeseries_records: list[dict] = []

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            inventory = ckpt["inventory"]
            timeseries_records = ckpt["timeseries_records"]
            completed_dates = ckpt["completed_dates"]
            log.info(
                "Resuming from checkpoint: %d/%d steps already completed",
                len(completed_dates),
                len(step_dates),
            )
        except Exception as e:
            log.warning("Failed to load checkpoint, starting fresh: %s", e)

    for i, step_date in enumerate(
        tqdm(step_dates, desc=f"{zone_name} steps")
    ):
        if step_date in completed_dates:
            continue

        step_ts = np.datetime64(step_date)
        log.info(
            "Step %d/%d: %s", i + 1, len(step_dates), step_date.date()
        )

        # ── Clean stale variables from previous step ─────────────────
        ds = _clean_stale_variables(ds)

        # ── tau_slow pass ─────────────────────────────────────────────
        ds_result = compute_empirical_features(ds, step_ts, tau_slow)
        if ds_result is None:
            log.warning("Skipping step %s — empirical failed for tau_slow", step_date.date())
            continue
        ds = ds_result

        result_slow = predict_tracks(xgb_clf, paths_gdf, ds)
        accumulate_inventory(
            inventory, result_slow, step_date, "slow", detection_threshold
        )

        # Timeseries record for slow
        n_det_slow = int((result_slow["p_debris"] >= timeseries_threshold).sum())
        mean_p_slow = float(result_slow["p_debris"].mean())
        max_p_slow = float(result_slow["p_debris"].max())
        timeseries_records.append(
            {
                "date": step_date,
                "tau": tau_slow,
                "n_track_detections": n_det_slow,
                "total_debris_area_m2": _estimate_debris_area(
                    result_slow, timeseries_threshold
                ),
                "mean_p_debris": mean_p_slow,
                "max_p_debris": max_p_slow,
            }
        )
        log.info(
            "  tau=%dd: %d detections (>%.1f), mean_p=%.3f, max_p=%.3f",
            tau_slow, n_det_slow, timeseries_threshold, mean_p_slow, max_p_slow,
        )

        # ── tau_fast pass (with slow as prior) ────────────────────────
        slow_priors = result_slow["p_debris"].values
        ds_result = compute_empirical_features(ds, step_ts, tau_fast)
        if ds_result is None:
            log.warning("Skipping fast pass for step %s", step_date.date())
            del result_slow
            gc.collect()
            continue
        ds = ds_result

        result_fast = predict_tracks(
            xgb_clf, paths_gdf, ds,
            prior_estimate=slow_priors,
        )
        accumulate_inventory(
            inventory, result_fast, step_date, "fast", detection_threshold
        )

        # Timeseries record for fast
        n_det_fast = int((result_fast["p_debris"] >= timeseries_threshold).sum())
        mean_p_fast = float(result_fast["p_debris"].mean())
        max_p_fast = float(result_fast["p_debris"].max())
        timeseries_records.append(
            {
                "date": step_date,
                "tau": tau_fast,
                "n_track_detections": n_det_fast,
                "total_debris_area_m2": _estimate_debris_area(
                    result_fast, timeseries_threshold
                ),
                "mean_p_debris": mean_p_fast,
                "max_p_debris": max_p_fast,
            }
        )
        log.info(
            "  tau=%dd: %d detections (>%.1f), mean_p=%.3f, max_p=%.3f",
            tau_fast, n_det_fast, timeseries_threshold, mean_p_fast, max_p_fast,
        )

        # Checkpoint and cleanup
        completed_dates.add(step_date)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "inventory": inventory,
                    "timeseries_records": timeseries_records,
                    "completed_dates": completed_dates,
                },
                f,
            )

        del result_slow, result_fast
        gc.collect()

    # ── 5. Build outputs ──────────────────────────────────────────────────
    log.info("Building season inventory and timeseries")
    inventory_gdf = build_inventory_gdf(inventory, paths_gdf, detection_threshold)
    timeseries_df = build_timeseries_df(timeseries_records)

    # ── 6. Save outputs ───────────────────────────────────────────────────
    season_label = season_start[:4] + "_" + season_end[:4]
    inv_path = cache_dir / f"{safe_name}_{season_label}_inventory.gpkg"
    ts_path = cache_dir / f"{safe_name}_{season_label}_timeseries.parquet"

    inventory_gdf.to_file(inv_path, driver="GPKG")
    log.info("Saved inventory → %s (%d tracks)", inv_path, len(inventory_gdf))

    timeseries_df.to_parquet(ts_path, index=False)
    log.info("Saved timeseries → %s (%d rows)", ts_path, len(timeseries_df))

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Removed checkpoint file")

    return inventory_gdf, timeseries_df


def _estimate_debris_area(
    result_gdf: gpd.GeoDataFrame, threshold: float
) -> float:
    """Estimate total debris area from tracks above threshold (no CNN seg)."""
    return float(
        result_gdf.loc[result_gdf["p_debris"] >= threshold, "geometry"]
        .area.sum()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual_tau_season_runner",
        description="Run dual-tau season analysis for avalanche detection",
    )
    parser.add_argument(
        "--zone",
        required=True,
        help='SNFAC zone name (e.g. "Sawtooth")',
    )
    parser.add_argument(
        "--season",
        required=True,
        help='Winter season as "YYYY-YYYY" (e.g. "2024-2025")',
    )
    parser.add_argument(
        "--tau-slow",
        type=float,
        default=24,
        help="Slow temporal decay tau in days (default: 24)",
    )
    parser.add_argument(
        "--tau-fast",
        type=float,
        default=3,
        help="Fast temporal decay tau in days (default: 3)",
    )
    parser.add_argument(
        "--out-dir",
        default="./local/issw/dual_tau_output/",
        metavar="DIR",
        help="Output directory (default: ./local/issw/dual_tau_output/)",
    )
    parser.add_argument(
        "--existing-runs-dir",
        default="./local/issw/high_danger_output/sarvalanche_runs/",
        metavar="DIR",
        help="Path to existing sarvalanche_runs/ dir to reuse FlowPy .gpkg and static .nc",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="CRS for processing (default: EPSG:4326)",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for counting a detection (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print step dates and data info without running inference",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse season
    try:
        start_year, end_year = args.season.split("-")
        season_start = f"{start_year}-11-01"
        season_end = f"{end_year}-04-30"
    except ValueError:
        parser.error('--season must be "YYYY-YYYY" (e.g. "2024-2025")')

    # Fetch zone geometry
    zones = fetch_snfac_zones()
    if args.zone not in zones:
        available = ", ".join(sorted(zones.keys()))
        parser.error(f'Zone "{args.zone}" not found. Available: {available}')

    zone_info = zones[args.zone]
    aoi = zone_info["geometry"]
    safe_name = args.zone.replace(" ", "_").replace("/", "-")

    # Set up zone cache directory
    zone_cache = out_dir / safe_name
    zone_cache.mkdir(parents=True, exist_ok=True)

    # Find existing static files from prior runs
    existing_dir = Path(args.existing_runs_dir)
    static_fp = None
    track_gpkg = None

    _REQUIRED_STATIC = {"dem", "slope", "aspect", "fcf"}

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

    # Run
    inventory_gdf, timeseries_df = run_dual_tau_season(
        aoi=aoi,
        zone_name=args.zone,
        season_start=season_start,
        season_end=season_end,
        cache_dir=zone_cache,
        tau_slow=args.tau_slow,
        tau_fast=args.tau_fast,
        crs=args.crs,
        static_fp=static_fp,
        track_gpkg=track_gpkg,
        detection_threshold=args.detection_threshold,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nInventory: {len(inventory_gdf)} tracks")
        if len(inventory_gdf) > 0:
            n_active = inventory_gdf["ran_this_season"].sum()
            print(f"  Active this season: {n_active}")
            print(
                f"  Max fast confidence: {inventory_gdf['fast_confidence_max'].max():.3f}"
            )
        print(f"\nTimeseries: {len(timeseries_df)} rows")
        if len(timeseries_df) > 0:
            for tau_val in timeseries_df["tau"].unique():
                subset = timeseries_df[timeseries_df["tau"] == tau_val]
                print(
                    f"  tau={tau_val}: {len(subset)} steps, "
                    f"max detections={subset['n_track_detections'].max()}"
                )


if __name__ == "__main__":
    main()
