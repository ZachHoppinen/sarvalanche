"""Evaluate v3 model against AKDOT/AKRR observed avalanche paths.

Runs season inference, then computes path-level metrics:
  - Detection rate at thresholds 0.2 and 0.5
  - Path-level F1, FPR at each threshold
  - D-size breakdown

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/evaluate_vs_akdot.py \
        --weights local/cnfaic/v3_experiment/patches/2025-12-15/v3_best.pt \
        [--nc local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc] \
        [--hrrr local/cnfaic/hrrr_temperature_2526.nc]
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rioxarray  # noqa: F401
import torch
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT
from sarvalanche.ml.v3.model import SinglePairDetector
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_season_pairs,
    normalize_dem_patch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
NC = ROOT / "local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc"
AKDOT_OBS = ROOT / "local/cnfaic/reported/akdot/akdot_avalanche_observations.csv"
AKDOT_PATHS = ROOT / "local/cnfaic/reported/akdot/avy_path_frequency.gpkg"
AKRR_OBS = ROOT / "local/cnfaic/reported/akrr/akrr_avalanche_obs.csv"
AKRR_PATHS = ROOT / "local/cnfaic/reported/akrr/akrr_avalanche_paths.gpkg"
HRRR = ROOT / "local/cnfaic/hrrr_temperature_2526.nc"

# No temporal split — all AKDOT data is validation (held out from training entirely)
THRESHOLDS = [0.2, 0.5]


# ── Inference ─────────────────────────────────────────────────────────

def sliding_window_inference(sar_scene, static_scene, model, device,
                              patch_size=V3_PATCH_SIZE, stride=32, batch_size=16):
    """Run CNN on sliding windows, average overlapping predictions."""
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            patches = []
            for y0, x0 in batch_coords:
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                static_patch = normalize_dem_patch(static_patch)
                patches.append(np.concatenate([sar_patch, static_patch], axis=0))

            batch_tensor = torch.from_numpy(np.stack(patches)).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    return np.where(count > 0, prob_sum / count, 0.0).astype(np.float32)


def aggregate_pair_probs(pair_probs, method="max"):
    """Aggregate per-pair probability maps into a single map per date.

    NaN = no SAR coverage. Uses nanmax so pixels with at least one
    covered pair get a real value; fully uncovered pixels stay NaN.
    """
    if len(pair_probs) == 0:
        return None
    with np.errstate(all="ignore"):
        if method == "max":
            return np.nanmax(pair_probs, axis=0)
        return np.nanmean(pair_probs, axis=0)


# ── Observation loading ───────────────────────────────────────────────

def load_akdot_obs(scene_box):
    """Load AKDOT observations within the scene bounds for 2025-2026 season."""
    obs_all = pd.read_csv(str(AKDOT_OBS))
    obs_all["avalanche_date"] = pd.to_datetime(obs_all["avalanche_date"])
    obs_2526 = obs_all[
        (obs_all["avalanche_date"] >= "2025-11-01")
        & (obs_all["avalanche_date"] <= "2026-04-30")
    ]
    obs_2526 = obs_2526[obs_2526["latitude"].notna()].copy()
    gdf = gpd.GeoDataFrame(
        obs_2526,
        geometry=gpd.points_from_xy(obs_2526.longitude, obs_2526.latitude),
        crs="EPSG:4326",
    )
    gdf = gdf[gdf.within(scene_box)]
    log.info("AKDOT 2025-2026: %d obs in scene", len(gdf))
    return gdf


def load_akdot_paths(scene_crs):
    """Load AKDOT avalanche path polygons."""
    paths = gpd.read_file(AKDOT_PATHS)
    if paths.crs != scene_crs:
        paths = paths.to_crs(scene_crs)
    return paths


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_vs_observations(
    date_prob_maps, obs_gdf, akdot_paths, ds, thresholds=THRESHOLDS,
):
    """Compute path-level F1 and detection rates.

    For each SAR date × path combination:
      - has_obs: observation within ±6 days on that path
      - has_detection: any pixel in path above threshold

    Returns DataFrame of per-obs results and path-date results.
    """
    scene_crs = ds.rio.crs
    H, W = ds.sizes["y"], ds.sizes["x"]
    bounds = ds.rio.bounds()
    transform = from_bounds(*bounds, W, H)

    # Convert paths to scene CRS if needed
    paths_proj = akdot_paths
    if paths_proj.crs != scene_crs:
        paths_proj = paths_proj.to_crs(scene_crs)

    # ── Per-observation detection rates ──
    obs_results = []
    val_obs = obs_gdf.copy()
    log.info("Validation obs: %d (all AKDOT, held out from training)", len(val_obs))

    for _, row in val_obs.iterrows():
        obs_date = row["avalanche_date"]
        obs_point = row.geometry

        # Find containing path
        containing = paths_proj[paths_proj.contains(obs_point)]
        if len(containing) == 0:
            # Try 150m buffer
            buf = paths_proj.copy()
            buf["geometry"] = buf.geometry.buffer(150)
            containing = buf[buf.contains(obs_point)]
        if len(containing) == 0:
            continue

        path_geom = containing.iloc[0].geometry
        path_name = containing.iloc[0].get("name", "unknown")

        try:
            path_mask = ~rasterio.features.geometry_mask(
                [path_geom], out_shape=(H, W), transform=transform, all_touched=True,
            )
        except Exception:
            continue
        if path_mask.sum() == 0:
            continue

        # Find closest date with CNN output
        cnn_dates = sorted(date_prob_maps.keys())
        best_date = None
        best_diff = float("inf")
        for d in cnn_dates:
            diff = abs((pd.Timestamp(d) - obs_date).days)
            if diff < best_diff:
                best_diff = diff
                best_date = d
        if best_date is None or best_diff > 6:
            continue

        prob_map = date_prob_maps[best_date]
        path_probs = prob_map[path_mask]

        rec = {
            "obs_id": row.get("objectid", ""),
            "date": obs_date.strftime("%Y-%m-%d"),
            "path_name": path_name,
            "dsize": row.get("dsize", ""),
            "n_path_px": int(path_mask.sum()),
            "mean_prob": float(np.mean(path_probs)),
            "max_prob": float(np.max(path_probs)),
            "cnn_date": best_date,
            "time_gap_days": best_diff,
        }
        for thresh in thresholds:
            rec[f"n_above_{thresh}"] = int((path_probs > thresh).sum())
            rec[f"detected_{thresh}"] = int((path_probs > thresh).sum()) > 0
        obs_results.append(rec)

    obs_df = pd.DataFrame(obs_results)

    # ── Path-level F1 (all paths × all dates) ──
    f1_results = []
    for cnn_date_str, prob_map in date_prob_maps.items():
        cnn_date = pd.Timestamp(cnn_date_str)
        # Skip pre-season dates
        if cnn_date < pd.Timestamp("2025-11-01"):
            continue

        for _, path_row in paths_proj.iterrows():
            path_name = path_row.get("name", "unknown")
            try:
                path_mask = ~rasterio.features.geometry_mask(
                    [path_row.geometry], out_shape=(H, W), transform=transform, all_touched=True,
                )
            except Exception:
                continue
            if path_mask.sum() == 0:
                continue

            path_probs = prob_map[path_mask]

            # Was there an observation on this path within ±6 days?
            has_obs = False
            nearby = val_obs[np.abs((val_obs["avalanche_date"] - cnn_date).dt.days) <= 6]
            for _, obs_row in nearby.iterrows():
                if path_row.geometry.contains(obs_row.geometry):
                    has_obs = True
                    break

            for thresh in thresholds:
                has_det = int((path_probs > thresh).sum()) > 0
                f1_results.append({
                    "cnn_date": cnn_date_str,
                    "path_name": path_name,
                    "threshold": thresh,
                    "has_obs": has_obs,
                    "has_detection": has_det,
                    "n_above": int((path_probs > thresh).sum()),
                })

    f1_df = pd.DataFrame(f1_results)
    return obs_df, f1_df


def print_report(obs_df, f1_df):
    """Print evaluation report."""
    print(f"\n{'='*70}")
    print("V3 EVALUATION vs AKDOT OBSERVATIONS")
    print(f"{'='*70}")

    if len(obs_df) == 0:
        print("No observations matched!")
        return

    # ── Per-observation detection rates ──
    print(f"\nMatched observations: {len(obs_df)}")
    for thresh in THRESHOLDS:
        col = f"detected_{thresh}"
        n_det = obs_df[col].sum()
        print(f"  Detection @{thresh}: {n_det}/{len(obs_df)} ({100*n_det/len(obs_df):.1f}%)")

    # ── D-size breakdown ──
    print(f"\n{'─'*60}")
    print("DETECTION BY D-SIZE")
    print(f"{'─'*60}")
    print(f"  {'D-size':<8} {'N_obs':>6} {'Det@0.2':>10} {'Det@0.5':>10}")
    for dsize in sorted(obs_df["dsize"].fillna("").astype(str).unique()):
        sub = obs_df[obs_df["dsize"].fillna("").astype(str) == dsize]
        n = len(sub)
        n20 = sub["detected_0.2"].sum()
        n50 = sub["detected_0.5"].sum()
        r20 = f"{100*n20/max(n,1):.0f}%"
        r50 = f"{100*n50/max(n,1):.0f}%"
        print(f"  {str(dsize):<8} {n:>6} {n20:>4}/{n} {r20:>4}  {n50:>4}/{n} {r50:>4}")
    n = len(obs_df)
    n20 = obs_df["detected_0.2"].sum()
    n50 = obs_df["detected_0.5"].sum()
    print(f"  {'ALL':<8} {n:>6} {n20:>4}/{n} {100*n20/max(n,1):.0f}%  {n50:>4}/{n} {100*n50/max(n,1):.0f}%")

    # ── Path-level F1 ──
    if len(f1_df) > 0:
        print(f"\n{'='*70}")
        print("PATH-LEVEL F1 (path × date combinations, val period only)")
        print(f"{'='*70}")
        for thresh in sorted(f1_df["threshold"].unique()):
            sub = f1_df[f1_df["threshold"] == thresh]
            tp = ((sub["has_obs"]) & (sub["has_detection"])).sum()
            fp = ((~sub["has_obs"]) & (sub["has_detection"])).sum()
            fn = ((sub["has_obs"]) & (~sub["has_detection"])).sum()
            tn = ((~sub["has_obs"]) & (~sub["has_detection"])).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            fpr = fp / max(fp + tn, 1)
            print(f"  Threshold > {thresh}:")
            print(f"    TP={tp:>5d}  FP={fp:>5d}  FN={fn:>5d}  TN={tn:>5d}")
            print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
            print(f"    FPR={fpr:.4f} ({fp} false path-date detections)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 model vs AKDOT observations")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--nc", type=Path, default=NC)
    parser.add_argument("--hrrr", type=Path, default=HRRR)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--pair-cache", type=Path, default=None,
                        help="Path to cached pair inference .npz. Skips CNN inference if exists.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # Load model
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    in_ch = ckpt["enc1.block.0.weight"].shape[1]
    model = SinglePairDetector(in_ch=in_ch, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt)
    model_sar_ch = in_ch - N_STATIC
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded model: %d params, in_ch=%d (sar=%d, static=%d)",
             n_params, in_ch, model_sar_ch, N_STATIC)

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    scene_box = box(*ds.rio.bounds())

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR from %s", args.hrrr)

    # Build static stack
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("Static stack shape: %s", static_scene.shape)

    # Get all season pairs
    log.info("Extracting season pairs...")
    pairs = get_all_season_pairs(ds, max_span_days=args.max_span_days, hrrr_ds=hrrr_ds)
    log.info("%d pairs extracted", len(pairs))

    # Run inference on all pairs — keep per-pair results
    out_dir = args.weights.parent
    cache_path = args.pair_cache or (out_dir / "v3_pair_inference.npz")

    if cache_path.exists():
        log.info("Loading cached pair inference from %s", cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        pair_probs = [p.astype(np.float32) for p in cache["pair_probs"]]
        pair_meta = [dict(m) for m in cache["pair_meta"]]
        log.info("  %d pairs loaded from cache", len(pair_probs))
    else:
        log.info("Running inference on all pairs...")
        pair_probs = []
        pair_meta = []
        for pi, pair in enumerate(pairs):
            ts = str(pair["t_start"])[:10]
            te = str(pair["t_end"])[:10]

            if (pi + 1) % 20 == 0 or pi == 0:
                log.info("  [%d/%d] %s → %s (track %s, %dd)",
                         pi + 1, len(pairs), ts, te, pair["track"], pair["span_days"])

            prob_map = sliding_window_inference(
                pair["sar"], static_scene, model, device,
                stride=args.stride, batch_size=args.batch_size,
            )

            # Mask out areas with no SAR coverage
            no_coverage = np.abs(pair["sar"][0]) < 1e-6
            prob_map[no_coverage] = np.nan

            pair_probs.append(prob_map)
            pair_meta.append({
                "t_start": str(pair["t_start"]),
                "t_end": str(pair["t_end"]),
                "span_days": int(pair["span_days"]),
                "track": str(pair["track"]),
            })

        # Save cache
        log.info("Saving pair inference cache to %s", cache_path)
        np.savez_compressed(
            cache_path,
            pair_probs=np.array(pair_probs, dtype=object),
            pair_meta=np.array(pair_meta, dtype=object),
        )
        log.info("  Saved (%d pairs)", len(pair_probs))

    # Load observations
    obs_gdf = load_akdot_obs(scene_box)
    akdot_paths = load_akdot_paths(ds.rio.crs)

    # ── Run pair-aware temporal onset ─────────────────────────────────
    from sarvalanche.ml.v3.temporal_onset import run_pair_temporal_onset

    onset_result, onset_dates_list, detected_cube = run_pair_temporal_onset(
        pair_probs, pair_meta,
        threshold=0.2,
        min_dates=2,
        hrrr_ds=hrrr_ds,
        coords={"y": ds.y.values, "x": ds.x.values},
        crs=str(ds.rio.crs) if ds.rio.crs else None,
    )

    # Save
    onset_path = out_dir / "v3_temporal_onset.nc"
    onset_result.to_netcdf(onset_path)
    log.info("Saved onset: %s", onset_path)

    # ── Evaluate against AKDOT observations ───────────────────────────
    H, W = ds.sizes["y"], ds.sizes["x"]
    scene_crs = ds.rio.crs
    transform = from_bounds(*ds.rio.bounds(), W, H)
    paths_proj = akdot_paths
    if paths_proj.crs != scene_crs:
        paths_proj = paths_proj.to_crs(scene_crs)

    candidate_mask = onset_result["candidate_mask"].values.astype(bool)
    # appearance_date is (event, y, x) — flatten all events for matching
    all_appearance = onset_result["appearance_date"].values  # (MAX_EVENTS, H, W)
    confidence = onset_result["confidence"].values
    peak_prob_arr = onset_result["peak_prob"].values
    n_dates_det = onset_result["n_dates_clean"].values

    val_obs = obs_gdf.copy()
    ONSET_WINDOW = 7  # days (AKDOT timestamps have time components, need margin)

    log.info("Evaluating %d val observations against %d candidates",
             len(val_obs), candidate_mask.sum())

    onset_obs_results = []
    for _, row in val_obs.iterrows():
        obs_date = row["avalanche_date"]
        obs_point = row.geometry

        containing = paths_proj[paths_proj.contains(obs_point)]
        if len(containing) == 0:
            buf = paths_proj.to_crs("EPSG:3338").copy()
            buf["geometry"] = buf.geometry.buffer(150)
            buf = buf.to_crs(scene_crs)
            containing = buf[buf.contains(obs_point)]
        if len(containing) == 0:
            continue

        path_geom = containing.iloc[0].geometry
        path_name = containing.iloc[0].get("name", "unknown")

        try:
            path_mask = ~rasterio.features.geometry_mask(
                [path_geom], out_shape=(H, W), transform=transform, all_touched=True,
            )
        except Exception:
            continue
        if path_mask.sum() == 0:
            continue

        in_path_cand = candidate_mask & path_mask
        n_cand = int(in_path_cand.sum())

        # Check onset date match
        onset_match = False
        if n_cand > 0:
            # Check all events at candidate pixels
            path_onsets = all_appearance[:, in_path_cand].ravel()
            valid = path_onsets[~np.isnat(path_onsets)]
            if len(valid) > 0:
                obs_dt64 = np.datetime64(obs_date)
                diffs = np.abs((valid - obs_dt64) / np.timedelta64(1, "D"))
                if diffs.min() <= ONSET_WINDOW:
                    onset_match = True

        path_conf = confidence[path_mask]
        path_peak = peak_prob_arr[path_mask]
        path_ndates = n_dates_det[path_mask]

        onset_obs_results.append({
            "obs_id": row.get("objectid", ""),
            "date": obs_date.strftime("%Y-%m-%d"),
            "path_name": path_name,
            "dsize": row.get("dsize", ""),
            "n_candidate_px": n_cand,
            "onset_match": onset_match,
            "detected": n_cand > 0 and onset_match,
            "detected_spatial": n_cand > 0,
            "max_peak_prob": float(np.nanmax(path_peak)) if len(path_peak) > 0 else 0,
            "mean_confidence": float(np.nanmean(path_conf[path_conf > 0])) if (path_conf > 0).any() else 0,
            "max_n_dates": int(np.nanmax(path_ndates)) if len(path_ndates) > 0 else 0,
        })

    onset_df = pd.DataFrame(onset_obs_results)

    # Path-level F1: for each path, does it have a candidate with onset
    # matching any observation date?
    f1_results = []
    for _, path_row in paths_proj.iterrows():
        path_name = path_row.get("name", "unknown")
        try:
            path_mask = ~rasterio.features.geometry_mask(
                [path_row.geometry], out_shape=(H, W), transform=transform, all_touched=True,
            )
        except Exception:
            continue
        if path_mask.sum() == 0:
            continue

        in_path_cand = candidate_mask & path_mask
        if in_path_cand.sum() == 0:
            # No detection on this path at any date
            has_any_obs = len(val_obs[val_obs.within(path_row.geometry)]) > 0
            f1_results.append({
                "path_name": path_name,
                "has_obs": has_any_obs,
                "has_detection": False,
            })
            continue

        # Get all onset dates in this path
        path_onsets = onset_dates_arr[in_path_cand]
        valid_onsets = path_onsets[~np.isnat(path_onsets)]

        # Check if any observation falls within ±ONSET_WINDOW of any onset
        has_obs = False
        has_matching_det = False
        for _, obs_row in val_obs.iterrows():
            if path_row.geometry.contains(obs_row.geometry):
                has_obs = True
                if len(valid_onsets) > 0:
                    obs_dt64 = np.datetime64(obs_row["avalanche_date"])
                    diffs = np.abs((valid_onsets - obs_dt64) / np.timedelta64(1, "D"))
                    if diffs.min() <= ONSET_WINDOW:
                        has_matching_det = True

        # Detection = candidates exist in path (regardless of obs match)
        has_detection = in_path_cand.sum() > 0

        f1_results.append({
            "path_name": path_name,
            "has_obs": has_obs,
            "has_detection": has_detection,
        })

    f1_df = pd.DataFrame(f1_results)

    # ── Print report ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("V3 PAIR-AWARE TEMPORAL ONSET EVALUATION")
    print(f"{'='*70}")

    if len(onset_df) > 0:
        n = len(onset_df)
        n_spatial = onset_df["detected_spatial"].sum()
        n_onset = onset_df["detected"].sum()
        print(f"\nMatched observations: {n}")
        print(f"  Spatial detection (candidate pixels in path): {n_spatial}/{n} ({100*n_spatial/n:.1f}%)")
        print(f"  + Onset match (±{ONSET_WINDOW}d): {n_onset}/{n} ({100*n_onset/n:.1f}%)")

        print(f"\n{'─'*60}")
        print("DETECTION BY D-SIZE")
        print(f"{'─'*60}")
        print(f"  {'D-size':<8} {'N_obs':>6} {'Spatial':>10} {'+ Onset':>10}")
        for dsize in sorted(onset_df["dsize"].fillna("").astype(str).unique()):
            sub = onset_df[onset_df["dsize"].fillna("").astype(str) == dsize]
            ns = sub["detected_spatial"].sum()
            no_ = sub["detected"].sum()
            nn = len(sub)
            print(f"  {str(dsize):<8} {nn:>6} {ns:>4}/{nn} {100*ns/max(nn,1):>3.0f}%  {no_:>4}/{nn} {100*no_/max(nn,1):>3.0f}%")
        ns = onset_df["detected_spatial"].sum()
        no_ = onset_df["detected"].sum()
        print(f"  {'ALL':<8} {n:>6} {ns:>4}/{n} {100*ns/n:.0f}%  {no_:>4}/{n} {100*no_/n:.0f}%")

        # D2+ with known dsize
        known = onset_df[onset_df["dsize"].fillna("").astype(str).str.match(r"^D\d")]
        d2plus = known[known["dsize"].astype(str).isin(["D2", "D2.5", "D3"])]
        if len(known) > 0:
            nk = len(known)
            nks = known["detected_spatial"].sum()
            nko = known["detected"].sum()
            print(f"\n  Known D-size only:")
            print(f"  {'ALL known':<12} {nk:>4}  spatial={nks}/{nk} ({100*nks/nk:.0f}%)  onset={nko}/{nk} ({100*nko/nk:.0f}%)")
        if len(d2plus) > 0:
            nd = len(d2plus)
            nds = d2plus["detected_spatial"].sum()
            ndo = d2plus["detected"].sum()
            print(f"  {'D2+':<12} {nd:>4}  spatial={nds}/{nd} ({100*nds/nd:.0f}%)  onset={ndo}/{nd} ({100*ndo/nd:.0f}%)")

    if len(f1_df) > 0:
        print(f"\n{'='*70}")
        print("PATH-LEVEL F1 (per-path, any date)")
        print(f"{'='*70}")
        tp = ((f1_df["has_obs"]) & (f1_df["has_detection"])).sum()
        fp = ((~f1_df["has_obs"]) & (f1_df["has_detection"])).sum()
        fn = ((f1_df["has_obs"]) & (~f1_df["has_detection"])).sum()
        tn = ((~f1_df["has_obs"]) & (~f1_df["has_detection"])).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr = fp / max(fp + tn, 1)
        print(f"    TP={tp:>5d}  FP={fp:>5d}  FN={fn:>5d}  TN={tn:>5d}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        print(f"    FPR={fpr:.4f} ({fp} false path detections)")

    # Save
    if len(onset_df) > 0:
        onset_df.to_csv(out_dir / "v3_onset_eval.csv", index=False)
    if len(f1_df) > 0:
        f1_df.to_csv(out_dir / "v3_f1_eval.csv", index=False)
    log.info("Saved all results to %s", out_dir)

    if hrrr_ds is not None:
        hrrr_ds.close()


if __name__ == "__main__":
    main()
