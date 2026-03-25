"""Evaluate v4 multi-scale model against SNFAC observations.

Runs per-pair inference with 3-scale inputs (fine, local context, regional),
pair-aware temporal onset, and compares against SNFAC observations using
FlowPy-modeled paths.

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v4/evaluate_vs_snfac.py \
        --weights local/issw/snfac/v3_experiment/v4_snfac_best.pt \
        --nc "local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --obs local/issw/snfac/snfac_obs_2021_2025.csv \
        --flowpy-paths local/issw/snfac/observations/all_flowpy_paths.gpkg \
        --zone "Sawtooth"
"""

import argparse
import ast
import logging
import time as _time
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
from skimage.transform import resize

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT, N_SAR, N_STATIC
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    get_all_season_pairs,
    normalize_dem_patch,
)
from sarvalanche.ml.v4.inference import build_v4_inputs, build_regional
from sarvalanche.ml.v4.model import MultiScaleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
ONSET_WINDOW = 7


def _build_regional(pair, static_scene, target=V3_PATCH_SIZE):
    """Build regional input for a pair. Delegates to shared inference code.

    Recovers raw dB diffs from log1p-compressed SAR channels since
    get_all_season_pairs returns compressed values but regional builder
    needs raw diffs (to match training which downsamples then compresses).
    """
    # Undo log1p: sign(x)*log1p(|x|) → sign(x)*expm1(|x|)
    change_vv = pair["sar"][0]
    change_vh = pair["sar"][1]
    vv_diff = np.sign(change_vv) * np.expm1(np.abs(change_vv))
    vh_diff = np.sign(change_vh) * np.expm1(np.abs(change_vh))

    return build_regional(
        None, static_scene,
        anf_norm=pair["sar"][3],
        vv_diff=vv_diff,
        vh_diff=vh_diff,
        span_days=pair["span_days"],
        target=target,
    )


def _extract_patch(sar_scene, static_scene, y0, x0, size):
    """Extract a patch, padding at edges."""
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    y0c, x0c = max(y0, 0), max(x0, 0)
    y1, x1 = min(y0 + size, H), min(x0 + size, W)
    sar = sar_scene[:, y0c:y1, x0c:x1]
    static = static_scene[:, y0c:y1, x0c:x1].copy()
    static = normalize_dem_patch(static)
    ah, aw = y1 - y0c, x1 - x0c
    if ah < size or aw < size:
        sar_p = np.zeros((sar.shape[0], size, size), dtype=np.float32)
        sta_p = np.zeros((static.shape[0], size, size), dtype=np.float32)
        py, px = y0c - y0, x0c - x0
        sar_p[:, py:py+ah, px:px+aw] = sar
        sta_p[:, py:py+ah, px:px+aw] = static
        return np.concatenate([sar_p, sta_p], axis=0)
    return np.concatenate([sar, static], axis=0)


def sliding_window_inference_v4(sar_scene, static_scene, regional, model, device,
                                 patch_size=V3_PATCH_SIZE, stride=32, batch_size=64):
    """Run sliding window inference with 3-scale v4 inputs.

    Uses shared build_v4_inputs() for per-patch DEM normalization,
    exactly matching the training data pipeline.
    """
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    C = N_SAR + N_STATIC
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    regional_t = torch.from_numpy(regional[np.newaxis]).to(device)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            B = len(batch_coords)

            fine_arr = np.empty((B, C, patch_size, patch_size), dtype=np.float32)
            ctx_arr = np.empty((B, C, patch_size, patch_size), dtype=np.float32)

            for i, (y0, x0) in enumerate(batch_coords):
                fine, local_ctx = build_v4_inputs(sar_scene, static_scene, y0, x0, patch_size)
                fine_arr[i] = fine
                ctx_arr[i] = local_ctx

            fine_t = torch.from_numpy(fine_arr).to(device)
            ctx_t = torch.from_numpy(ctx_arr).to(device)
            reg_t = regional_t.expand(B, -1, -1, -1)

            logits = model(fine_t, ctx_t, reg_t)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    return np.where(count > 0, prob_sum / count, np.nan).astype(np.float32)


def parse_location_point(loc_str):
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate v4 vs SNFAC observations")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--obs", type=Path, default=ROOT / "local/issw/snfac/snfac_obs_2021_2025.csv")
    parser.add_argument("--flowpy-paths", type=Path, default=ROOT / "local/issw/snfac/observations/all_flowpy_paths.gpkg")
    parser.add_argument("--zone", type=str, default="Sawtooth")
    parser.add_argument("--season", type=str, default="2024-2025")
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-span-days", type=int, default=60)
    parser.add_argument("--hrrr", type=Path, default=None,
                        help="HRRR temperature NetCDF for melt filtering")
    parser.add_argument("--pair-cache", type=Path, default=None,
                        help="Path to .npz file. If exists, load pair probs from cache (skip inference). "
                             "If doesn't exist, run inference and save cache here.")
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

    # Load model — handle both old (bare state_dict) and new checkpoint formats
    raw = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        log.info("Checkpoint metadata: epoch=%s, val_loss=%s",
                 raw.get("epoch", "?"), raw.get("val_loss", "?"))
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw

    # Handle both old (encoder.enc1.block.0) and new (encoder.enc1.0.block.0) architectures
    for key in ["encoder.enc1.block.0.weight", "encoder.enc1.0.block.0.weight"]:
        if key in state_dict:
            in_ch = state_dict[key].shape[1]
            break
    else:
        in_ch = 11  # fallback
    model = MultiScaleDetector(in_ch=in_ch, base_ch=args.base_ch).to(device)
    model.load_state_dict(state_dict)
    log.info("Loaded v4 model: in_ch=%d, base_ch=%d, %d params",
             in_ch, args.base_ch, sum(p.numel() for p in model.parameters()))
    gate_val = model.fpn_gate.item()
    log.info("FPN gate: %.3f (sigmoid=%.4f)", gate_val, 1.0 / (1.0 + np.exp(-gate_val)))

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

    # ── Load or compute pair probabilities ────────────────────────────
    # Uses a directory of individual .npy files + metadata.json so we can:
    #   1. Save incrementally (resume if interrupted)
    #   2. Load one pair at a time (no 9GB in memory)
    import json

    cache_dir = args.pair_cache  # now a directory, not a file
    pair_metas = []

    if cache_dir and cache_dir.exists() and (cache_dir / "meta.json").exists():
        log.info("Loading pair cache from %s", cache_dir)
        with open(cache_dir / "meta.json") as f:
            pair_metas = json.load(f)
        log.info("  Found %d cached pairs", len(pair_metas))
    else:
        # Run inference, saving each pair to disk immediately
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        ps = V3_PATCH_SIZE
        n_patches_per_pair = ((H - ps) // args.stride + 1) * ((W - ps) // args.stride + 1)
        log.info("Running v4 inference: %d pairs × %d patches/pair (stride=%d, batch=%d)",
                 len(pairs), n_patches_per_pair, args.stride, args.batch_size)

        pair_metas = []
        t_start_all = _time.time()

        for pi, pair in enumerate(pairs):
            t_pair = _time.time()

            sar_scene = pair["sar"]  # (N_SAR, H, W)

            # Build regional for this pair
            regional = _build_regional(pair, static_scene)

            prob_map = sliding_window_inference_v4(
                sar_scene, static_scene, regional, model, device,
                stride=args.stride, batch_size=args.batch_size,
            )
            no_cov = np.abs(sar_scene[0]) < 1e-6
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

            meta = {
                "t_start": str(pair["t_start"]),
                "t_end": str(pair["t_end"]),
                "span_days": int(pair["span_days"]),
                "track": str(pair["track"]),
            }
            pair_metas.append(meta)

            # Save to disk immediately (float16 to halve storage)
            if cache_dir:
                t0 = str(pair["t_start"])[:10]
                t1 = str(pair["t_end"])[:10]
                trk = str(pair["track"])
                fname = f"trk{trk}_{t0}_{t1}_{meta['span_days']}d.npy"
                np.save(cache_dir / fname, prob_map.astype(np.float16))
                meta["cache_file"] = fname

        log.info("Inference done: %d pairs in %.1f min (%.1fs/pair avg)",
                 len(pairs), (_time.time() - t_start_all) / 60,
                 (_time.time() - t_start_all) / max(len(pairs), 1))

        if cache_dir:
            with open(cache_dir / "meta.json", "w") as f:
                json.dump(pair_metas, f)
            total_bytes = sum(f.stat().st_size for f in cache_dir.glob("pair_*.npy"))
            log.info("  Cache saved: %d pairs (%.1f GB) in %s",
                     len(pair_metas), total_bytes / 1e9, cache_dir)

    # ── Load pair probs one at a time for temporal onset ──────────────
    # The onset function needs all probs in memory, but as float16→float32
    # on load, we only need ~4.5GB instead of 9GB
    log.info("Loading pair probs for temporal onset...")
    pair_probs = []
    for pi, meta in enumerate(pair_metas):
        if cache_dir and "cache_file" in meta:
            prob = np.load(cache_dir / meta["cache_file"]).astype(np.float32)
        elif cache_dir and (cache_dir / f"pair_{pi:04d}.npy").exists():
            # Fallback for old-style numbered cache
            prob = np.load(cache_dir / f"pair_{pi:04d}.npy").astype(np.float32)
        else:
            raise RuntimeError(f"Missing pair prob {pi} and no cache")
        pair_probs.append(prob)

    from sarvalanche.ml.v3.temporal_onset import run_pair_temporal_onset
    onset_result, onset_dates, _ = run_pair_temporal_onset(
        pair_probs, pair_metas,
        threshold=0.2, min_dates=2,
        hrrr_ds=hrrr_ds,
        coords={"y": ds.y.values, "x": ds.x.values},
        crs=str(ds.rio.crs) if ds.rio.crs else None,
    )

    del pair_probs

    # Load observations
    obs_df = pd.read_csv(args.obs)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    season_parts = args.season.split("-")
    season_start = f"{season_parts[0]}-11-01"
    season_end = f"{season_parts[1]}-04-30"
    obs_season = obs_df[
        (obs_df["date"] >= season_start) & (obs_df["date"] <= season_end)
        & obs_df["zone_name"].str.contains(args.zone, na=False)
    ].copy()

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
    if fp_zone.crs and ds.rio.crs and str(fp_zone.crs) != str(ds.rio.crs):
        fp_zone = fp_zone.to_crs(ds.rio.crs)
    log.info("FlowPy paths: %d for %s (CRS: %s)", len(fp_zone), args.zone, fp_zone.crs)

    # Evaluate
    transform = from_bounds(*ds.rio.bounds(), W, H)
    candidate = onset_result["candidate_mask"].values.astype(bool)
    all_max_t_start = onset_result["max_t_start"].values   # (MAX_EVENTS, H, W) — start of likely event period
    all_min_t_end = onset_result["min_t_end"].values       # (MAX_EVENTS, H, W) — end of likely event period
    confidence = onset_result["confidence"].values

    results = []
    for _, row in obs_gdf.iterrows():
        obs_date = row["date"]
        obs_id = row["id"]
        obs_point = row.geometry

        matching = fp_zone[fp_zone["obs_id"] == obs_id]
        if len(matching) == 0:
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
            obs_dt64 = np.datetime64(obs_date)

            # For each event at each candidate pixel, check if obs falls
            # within [max_t_start, min_t_end] — the likely event window.
            # Distance = 0 if inside, positive if outside.
            best_dist = np.inf
            for ei in range(all_max_t_start.shape[0]):
                win_start_vals = all_max_t_start[ei][in_path]
                win_end_vals = all_min_t_end[ei][in_path]

                for pi in range(len(win_start_vals)):
                    ws = win_start_vals[pi]
                    we = win_end_vals[pi]
                    if np.isnat(ws):
                        continue

                    epoch = np.datetime64('1970-01-01', 'ns')
                    day = np.timedelta64(1, 'D')
                    obs_day = (obs_dt64 - epoch) / day
                    ws_day = (ws - epoch) / day

                    if np.isnat(we):
                        dist = abs(obs_day - ws_day)
                    else:
                        we_day = (we - epoch) / day
                        if obs_day < ws_day:
                            dist = ws_day - obs_day
                        elif obs_day > we_day:
                            dist = obs_day - we_day
                        else:
                            dist = 0.0

                    if dist < best_dist:
                        best_dist = dist

            if np.isfinite(best_dist):
                onset_diff_days = float(best_dist)
                if best_dist <= ONSET_WINDOW:
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
    print(f"V4 SNFAC EVALUATION: {args.zone} {args.season}")
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

    known = rdf[rdf["d_size"].notna()]
    d2plus = known[known["d_size"] >= 2.0]
    if len(known) > 0:
        nks = known["detected_spatial"].sum()
        nkd = known["detected"].sum()
        nk = len(known)
        print(f"\n  Known D-size: {nk}  spatial={nks}/{nk} ({100*nks/nk:.0f}%)  onset={nkd}/{nk} ({100*nkd/nk:.0f}%)")
    if len(d2plus) > 0:
        nds = d2plus["detected_spatial"].sum()
        ndd_val = d2plus["detected"].sum()
        nd2 = len(d2plus)
        print(f"  D2+:          {nd2}  spatial={nds}/{nd2} ({100*nds/nd2:.0f}%)  onset={ndd_val}/{nd2} ({100*ndd_val/nd2:.0f}%)")

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

            print(f"\n  Mean |onset diff| by D-size:")
            for ds_val in sorted(spatial["d_size"].dropna().unique()):
                sub = spatial[spatial["d_size"] == ds_val]["onset_diff_days"].dropna()
                if len(sub) > 0:
                    print(f"    D{ds_val}: {sub.abs().mean():.1f}d (n={len(sub)})")

    # Save
    out_dir = args.weights.parent
    rdf.to_csv(out_dir / f"v4_snfac_eval_{args.zone.lower().replace(' ','_')}.csv", index=False)
    onset_result.to_netcdf(out_dir / f"v4_snfac_onset_{args.zone.lower().replace(' ','_')}.nc")
    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
