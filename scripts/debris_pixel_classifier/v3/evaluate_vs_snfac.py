"""Evaluate v3 model against SNFAC observations.

Runs per-pair inference, pair-aware temporal onset, and compares
against SNFAC observations using FlowPy-modeled paths.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v3/evaluate_vs_snfac.py \
        --weights local/issw/v3_experiment/v3_snfac_best.pt \
        --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --obs local/issw/snfac_obs_2021_2025.csv \
        --flowpy-paths local/issw/observations/all_flowpy_paths.gpkg \
        --zone "Sawtooth"
"""

import argparse
import ast
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
from sarvalanche.ml.v3.channels import N_INPUT, N_STATIC
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
ONSET_WINDOW = 7


def _run_model_batch(patches_np, model, device):
    """Run model on a batch, return probabilities."""
    batch_tensor = torch.from_numpy(np.stack(patches_np)).to(device)
    logits = model(batch_tensor)
    return torch.sigmoid(logits).cpu().numpy()[:, 0]


# TTA transforms: (transform_fn, inverse_fn) pairs
# Must handle aspect channels: northing (idx 6) and easting (idx 7)
# hflip: east↔west (negate easting), vflip: north↔south (negate northing)
# rot90: northing→easting, easting→-northing
_N_IDX = 6  # aspect_northing in concatenated input
_E_IDX = 7  # aspect_easting in concatenated input


def _hflip_fwd(x):
    out = x[:, :, ::-1].copy()
    out[_E_IDX] = -out[_E_IDX]  # east↔west
    return out


def _vflip_fwd(x):
    out = x[:, ::-1, :].copy()
    out[_N_IDX] = -out[_N_IDX]  # north↔south
    return out


def _rot90_fwd(x):
    out = np.rot90(x, 1, (1, 2)).copy()
    # After spatial rot90: north→east, east→-north
    n_orig = out[_N_IDX].copy()
    out[_N_IDX] = out[_E_IDX]
    out[_E_IDX] = -n_orig
    return out


TTA_TRANSFORMS = [
    (lambda x: x, lambda x: x),                             # identity
    (_hflip_fwd, lambda x: x[:, ::-1].copy()),               # hflip
    (_vflip_fwd, lambda x: x[::-1, :].copy()),               # vflip
    (_rot90_fwd, lambda x: np.rot90(x, -1, (0, 1)).copy()),  # rot90
]


def sliding_window_inference(sar_scene, static_scene, model, device,
                              patch_size=V3_PATCH_SIZE, stride=32, batch_size=16,
                              model_sar_ch=None, tta=True):
    """Run sliding window CNN inference with optional TTA."""
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    transforms = TTA_TRANSFORMS if tta else TTA_TRANSFORMS[:1]  # identity only if no TTA

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]

            # Build base patches
            base_patches = []
            for y0, x0 in batch_coords:
                sar_patch = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                if model_sar_ch is not None and sar_patch.shape[0] > model_sar_ch:
                    sar_patch = sar_patch[:model_sar_ch]
                static_patch = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size].copy()
                static_patch = normalize_dem_patch(static_patch)
                base_patches.append(np.concatenate([sar_patch, static_patch], axis=0))

            # Run each TTA transform, average predictions
            avg_probs = np.zeros((len(base_patches), patch_size, patch_size), dtype=np.float64)
            for fwd_fn, inv_fn in transforms:
                aug_patches = [fwd_fn(p) for p in base_patches]
                probs = _run_model_batch(aug_patches, model, device)
                # Invert the transform on the output
                for i in range(len(probs)):
                    avg_probs[i] += inv_fn(probs[i])
            avg_probs /= len(transforms)

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += avg_probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1
    return np.where(count > 0, prob_sum / count, np.nan).astype(np.float32)


def parse_location_point(loc_str):
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 vs SNFAC observations")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--obs", type=Path, default=ROOT / "local/issw/snfac/snfac_obs_2021_2025.csv")
    parser.add_argument("--flowpy-paths", type=Path, default=ROOT / "local/issw/snfac/observations/all_flowpy_paths.gpkg")
    parser.add_argument("--zone", type=str, default="Sawtooth")
    parser.add_argument("--season", type=str, default="2024-2025")
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--hrrr", type=Path, default=None,
                        help="HRRR temperature NetCDF for melt filtering")
    parser.add_argument("--pair-cache", type=Path, default=None)
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

    # Load model — detect input channels from checkpoint
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    in_ch = ckpt["enc1.block.0.weight"].shape[1]
    model = SinglePairDetector(in_ch=in_ch, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt)
    model_sar_ch = in_ch - N_STATIC  # SAR channels the model expects
    log.info("Loaded model: in_ch=%d (sar=%d, static=%d)", in_ch, model_sar_ch, N_STATIC)
    log.info("Loaded model: %d params", sum(p.numel() for p in model.parameters()))

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    H, W = ds.sizes["y"], ds.sizes["x"]
    scene_box = box(*ds.rio.bounds())

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR from %s", args.hrrr)

    # Build static stack
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)
    log.info("Static stack: %s", static_scene.shape)

    # Get pairs
    pairs = get_all_season_pairs(ds, max_span_days=args.max_span_days, hrrr_ds=hrrr_ds)
    log.info("%d pairs", len(pairs))

    # Run inference — store per-pair prob maps (thresholded sparse)
    out_dir = args.weights.parent

    import time as _time
    H, W = ds.sizes["y"], ds.sizes["x"]
    ps = V3_PATCH_SIZE
    n_patches_per_pair = ((H - ps) // args.stride + 1) * ((W - ps) // args.stride + 1)
    log.info("Running inference: %d pairs × %d patches/pair (stride=%d, batch=%d)",
             len(pairs), n_patches_per_pair, args.stride, args.batch_size)

    pair_probs = []
    pair_metas = []
    t_start_all = _time.time()
    for pi, pair in enumerate(pairs):
        t_pair = _time.time()

        prob_map = sliding_window_inference(
            pair["sar"], static_scene, model, device,
            stride=args.stride, batch_size=args.batch_size,
            model_sar_ch=model_sar_ch,
            tta=False,
        )
        no_cov = np.abs(pair["sar"][0]) < 1e-6
        prob_map[no_cov] = np.nan

        n_above = np.nansum(prob_map > 0.2)
        elapsed_pair = _time.time() - t_pair
        elapsed_total = _time.time() - t_start_all
        pairs_done = pi + 1
        eta = elapsed_total / pairs_done * (len(pairs) - pairs_done)

        if pairs_done <= 3 or pairs_done % 10 == 0:
            log.info("  [%d/%d] %s→%s track %s %dd | %.1fs/pair | >0.2: %d px | ETA: %.0fm",
                     pairs_done, len(pairs),
                     str(pair["t_start"])[:10], str(pair["t_end"])[:10],
                     pair["track"], pair["span_days"],
                     elapsed_pair, int(n_above), eta / 60)

        pair_probs.append(prob_map)
        pair_metas.append({
            "t_start": str(pair["t_start"]),
            "t_end": str(pair["t_end"]),
            "span_days": int(pair["span_days"]),
            "track": str(pair["track"]),
        })

    log.info("Inference done: %d pairs in %.1f min (%.1fs/pair avg)",
             len(pairs), (_time.time() - t_start_all) / 60,
             (_time.time() - t_start_all) / max(len(pairs), 1))

    # Run temporal onset with multi-peak support
    from sarvalanche.ml.v3.temporal_onset import run_pair_temporal_onset
    onset_result, onset_dates, _ = run_pair_temporal_onset(
        pair_probs, pair_metas,
        threshold=0.2, min_dates=2,
        hrrr_ds=hrrr_ds,
        coords={"y": ds.y.values, "x": ds.x.values},
        crs=str(ds.rio.crs) if ds.rio.crs else None,
    )

    # Free pair probs
    del pair_probs

    # Load observations
    obs_df = pd.read_csv(args.obs)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    # Parse season
    season_parts = args.season.split("-")
    season_start = f"{season_parts[0]}-11-01"
    season_end = f"{season_parts[1]}-04-30"
    obs_season = obs_df[
        (obs_df["date"] >= season_start) & (obs_df["date"] <= season_end)
        & obs_df["zone_name"].str.contains(args.zone, na=False)
    ].copy()

    # Parse lat/lng
    lats, lngs = [], []
    for loc in obs_season["location_point"]:
        lat, lng = parse_location_point(loc)
        lats.append(lat)
        lngs.append(lng)
    obs_season = obs_season.copy()
    obs_season["lat"] = lats
    obs_season["lng"] = lngs
    obs_gdf = gpd.GeoDataFrame(
        obs_season, geometry=gpd.points_from_xy(obs_season.lng, obs_season.lat), crs="EPSG:4326",
    )
    obs_gdf = obs_gdf[obs_gdf.within(scene_box)]
    log.info("Observations: %d in scene (%s %s)", len(obs_gdf), args.zone, args.season)

    # Load FlowPy paths
    fp_all = gpd.read_file(args.flowpy_paths)
    fp_zone = fp_all[fp_all["zone_name"].str.contains(args.zone, na=False)].copy()
    # Reproject to scene CRS if needed
    if fp_zone.crs and ds.rio.crs and str(fp_zone.crs) != str(ds.rio.crs):
        fp_zone = fp_zone.to_crs(ds.rio.crs)
    log.info("FlowPy paths: %d for %s (CRS: %s)", len(fp_zone), args.zone, fp_zone.crs)

    # Evaluate
    transform = from_bounds(*ds.rio.bounds(), W, H)
    candidate = onset_result["candidate_mask"].values.astype(bool)
    all_appearance = onset_result["appearance_date"].values  # (MAX_EVENTS, H, W)
    confidence = onset_result["confidence"].values

    results = []
    for _, row in obs_gdf.iterrows():
        obs_date = row["date"]
        obs_id = row["id"]
        obs_point = row.geometry

        # Find FlowPy path for this observation
        matching = fp_zone[fp_zone["obs_id"] == obs_id]
        if len(matching) == 0:
            # Try spatial match — find paths that contain the obs point
            matching = fp_zone[fp_zone.contains(obs_point)]
        if len(matching) == 0:
            continue

        path_geom = matching.iloc[0].geometry
        try:
            path_mask = ~rasterio.features.geometry_mask(
                [path_geom], out_shape=(H, W), transform=transform, all_touched=True,
            )
        except Exception:
            continue
        if path_mask.sum() == 0:
            continue

        in_path = candidate & path_mask
        n_cand = int(in_path.sum())

        onset_match = False
        onset_diff_days = np.nan
        if n_cand > 0:
            # Check all events at candidate pixels
            path_app = all_appearance[:, in_path].ravel()
            valid = path_app[~np.isnat(path_app)]
            if len(valid) > 0:
                obs_dt64 = np.datetime64(obs_date)
                diffs = (valid - obs_dt64) / np.timedelta64(1, "D")
                abs_diffs = np.abs(diffs)
                closest_idx = abs_diffs.argmin()
                onset_diff_days = float(diffs[closest_idx])
                if abs_diffs[closest_idx] <= ONSET_WINDOW:
                    onset_match = True

        results.append({
            "obs_id": obs_id,
            "date": obs_date.strftime("%Y-%m-%d"),
            "location_name": row.get("location_name", ""),
            "d_size": row.get("d_size", np.nan),
            "n_candidate_px": n_cand,
            "detected_spatial": n_cand > 0,
            "onset_match": onset_match,
            "onset_diff_days": onset_diff_days,
            "detected": n_cand > 0 and onset_match,
        })

    rdf = pd.DataFrame(results)

    # Print report
    print(f"\n{'='*70}")
    print(f"V3 SNFAC EVALUATION: {args.zone} {args.season}")
    print(f"{'='*70}")

    if len(rdf) == 0:
        print("No matched observations!")
        return

    n = len(rdf)
    ns = rdf["detected_spatial"].sum()
    nd = rdf["detected"].sum()
    print(f"\nMatched observations: {n}")
    print(f"  Spatial detection: {ns}/{n} ({100*ns/n:.1f}%)")
    print(f"  + Onset match (±{ONSET_WINDOW}d): {nd}/{n} ({100*nd/n:.1f}%)")

    print(f"\n{'─'*60}")
    print("DETECTION BY D-SIZE")
    print(f"{'─'*60}")
    print(f"  {'D-size':<8} {'N':>5} {'Spatial':>10} {'Onset':>10}")
    for ds_val in sorted(rdf["d_size"].dropna().unique()):
        sub = rdf[rdf["d_size"] == ds_val]
        nss = sub["detected_spatial"].sum()
        ndd = sub["detected"].sum()
        nn = len(sub)
        print(f"  D{ds_val:<7} {nn:>4}  {nss:>4}/{nn} ({100*nss/max(nn,1):>3.0f}%)  {ndd:>4}/{nn} ({100*ndd/max(nn,1):>3.0f}%)")

    # Known D-size summaries
    known = rdf[rdf["d_size"].notna()]
    d2plus = known[known["d_size"] >= 2.0]
    if len(known) > 0:
        nks = known["detected_spatial"].sum()
        nkd = known["detected"].sum()
        nk = len(known)
        print(f"\n  Known D-size: {nk}  spatial={nks}/{nk} ({100*nks/nk:.0f}%)  onset={nkd}/{nk} ({100*nkd/nk:.0f}%)")
    if len(d2plus) > 0:
        nds = d2plus["detected_spatial"].sum()
        ndd = d2plus["detected"].sum()
        nd2 = len(d2plus)
        print(f"  D2+:          {nd2}  spatial={nds}/{nd2} ({100*nds/nd2:.0f}%)  onset={ndd}/{nd2} ({100*ndd/nd2:.0f}%)")

    # Onset match at multiple windows
    spatial = rdf[rdf["detected_spatial"]]
    if len(spatial) > 0:
        print(f"\n{'─'*60}")
        print("ONSET TIMING (spatially detected observations only)")
        print(f"{'─'*60}")
        valid_diffs = spatial["onset_diff_days"].dropna()
        if len(valid_diffs) > 0:
            abs_diffs = valid_diffs.abs()
            print(f"  Observations with appearance date: {len(valid_diffs)}/{len(spatial)}")
            print(f"  Mean onset diff: {valid_diffs.mean():+.1f} days (negative = detected before obs)")
            print(f"  Median onset diff: {valid_diffs.median():+.1f} days")
            print(f"  Mean |onset diff|: {abs_diffs.mean():.1f} days")
            print(f"  Std onset diff: {valid_diffs.std():.1f} days")
            print(f"\n  Match rate by window:")
            for window in [3, 7, 12, 18, 24, 30, 45, 60]:
                n_match = (abs_diffs <= window).sum()
                print(f"    ±{window:>2d} days: {n_match}/{len(valid_diffs)} ({100*n_match/len(valid_diffs):.0f}%)")

            # By D-size
            print(f"\n  Mean |onset diff| by D-size:")
            for ds_val in sorted(spatial["d_size"].dropna().unique()):
                sub = spatial[spatial["d_size"] == ds_val]["onset_diff_days"].dropna()
                if len(sub) > 0:
                    print(f"    D{ds_val}: {sub.abs().mean():.1f}d (n={len(sub)})")

    # Save
    rdf.to_csv(out_dir / f"v3_snfac_eval_{args.zone.lower().replace(' ','_')}.csv", index=False)
    onset_result.to_netcdf(out_dir / f"v3_snfac_onset_{args.zone.lower().replace(' ','_')}.nc")
    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
