"""AK-only experiment with HRRR melt filtering.

Compares human_only baseline (3ch, no HRRR) against melt-filtered variants:
  - human_only_3ch: baseline, no HRRR (existing weights if available)
  - human_only_4ch: human labels with melt_weight 4th SAR channel + filtered d_empirical
  - auto_melt_ft: auto-label with melt-filtered d_empirical → finetune on human (4ch)

All AK data only — no SNFAC pretrain.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

NC = Path("local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc")
HUMAN_LABELS = Path("local/cnfaic/debris_shapes")
AKDOT_OBS = Path("local/cnfaic/reported/akdot/akdot_avalanche_observations.csv")
AKDOT_PATHS = Path("local/cnfaic/reported/akdot/avy_path_frequency.gpkg")
AKRR_OBS = Path("local/cnfaic/reported/akrr/akrr_avalanche_obs.csv")
AKRR_PATHS = Path("local/cnfaic/reported/akrr/akrr_avalanche_paths.gpkg")
HRRR = Path("local/cnfaic/hrrr_temperature_2526.nc")
OUT = Path("local/cnfaic/melt_experiment")
TAU = 6


def train(data_dirs, out_weights, epochs=50, lr=1e-3, resume=None, sar_ch=3):
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]
    else:
        data_dirs = [Path(d) for d in data_dirs]
    cmd = [
        sys.executable, "scripts/debris_pixel_classifier/v2/train.py",
        "--data-dir", *[str(d) for d in data_dirs],
        "--epochs", str(epochs), "--lr", str(lr),
        "--batch-size", "4", "--out", str(out_weights),
        "--sar-channels", str(sar_ch),
    ]
    if resume and Path(resume).exists():
        cmd += ["--resume", str(resume)]
    log.info("  CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


def extract_patches(nc, labels_dir, out_dir, date, hrrr_path=None):
    gpkg = labels_dir / f"avalanche_labels_{date}.gpkg"
    if not gpkg.exists():
        return
    geotiff_dir = labels_dir / "geotiffs" / date
    gt_arg = ["--geotiff-dir", str(geotiff_dir)] if geotiff_dir.is_dir() and list(geotiff_dir.glob("*.tif")) else []
    cmd = [
        sys.executable, "scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py",
        "--nc", str(nc), "--polygons", str(gpkg),
        *gt_arg,
        "--date", date, "--tau", str(TAU),
        "--out-dir", str(out_dir / date),
        "--stride", "64", "--neg-ratio", "3.0",
        "--pairs", "--max-pairs", "4",
    ]
    if hrrr_path and Path(hrrr_path).exists():
        cmd += ["--hrrr", str(hrrr_path)]
    subprocess.run(cmd, check=False)


def run_inference(nc, weights, out_dir, sar_ch=3, hrrr_path=None):
    cmd = [
        sys.executable, "scripts/debris_pixel_classifier/v2/full_season_inference.py",
        "--nc", str(nc), "--weights", str(weights),
        "--season", "2025-2026", "--tau", str(TAU),
        "--out-dir", str(out_dir),
        "--no-tiffs", "--stride", "32", "--batch-size", "16",
        "--sar-channels", str(sar_ch),
        "--pairs", "--max-pairs", "4",
    ]
    if hrrr_path and Path(hrrr_path).exists():
        cmd += ["--hrrr", str(hrrr_path)]
    subprocess.run(cmd, check=False)


def count_patches(d):
    p = n = 0
    if not d.exists():
        return 0, 0
    for lf in d.rglob("labels.json"):
        with open(lf) as f:
            labels = json.load(f)
        for v in labels.values():
            if v.get("label") == 1: p += 1
            elif v.get("label") == 0: n += 1
    return p, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights_dir = OUT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Auto-labeling with melt-filtered d_empirical
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 1: Smart auto-labeling with HRRR melt filtering ===")

    from sarvalanche.io.dataset import load_netcdf_to_dataset
    ds = load_netcdf_to_dataset(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    hrrr_ds = xr.open_dataset(HRRR) if HRRR.exists() else None
    if hrrr_ds is not None:
        log.info("  Loaded HRRR from %s", HRRR)

    # Import auto-labeler
    from scripts.debris_pixel_classifier.v2.run_cnfaic_experiment import (
        smart_auto_label, _compute_empirical, build_glacier_mask,
    )

    glacier_mask = build_glacier_mask(ds)

    times = pd.DatetimeIndex(ds.time.values)
    in_season = times[(times.month >= 11) | (times.month <= 4)]
    all_dates = sorted(set(t.strftime("%Y-%m-%d") for t in in_season))

    # Select dates using AKDOT obs density
    scene_box = box(*ds.rio.bounds())
    obs_all = pd.read_csv(str(AKDOT_OBS))
    obs_all["avalanche_date"] = pd.to_datetime(obs_all["avalanche_date"])
    obs_2526 = obs_all[(obs_all["avalanche_date"] >= "2025-11-01") & (obs_all["avalanche_date"] <= "2026-04-30")]
    obs_2526 = obs_2526[obs_2526["latitude"].notna()].copy()
    obs_gdf = gpd.GeoDataFrame(
        obs_2526, geometry=gpd.points_from_xy(obs_2526.longitude, obs_2526.latitude), crs="EPSG:4326",
    )
    obs_gdf = obs_gdf[obs_gdf.within(scene_box)]

    date_scores = []
    for d in all_dates:
        t = pd.Timestamp(d)
        nearby = obs_gdf[(obs_gdf["avalanche_date"] >= t - pd.Timedelta(days=6)) &
                          (obs_gdf["avalanche_date"] <= t + pd.Timedelta(days=6))]
        date_scores.append({"date": d, "n_obs": len(nearby)})
    score_df = pd.DataFrame(date_scores)

    selected = set()
    used = set()
    def ok(d, gap=12):
        return all(abs((pd.Timestamp(d) - pd.Timestamp(s)).days) >= gap for s in used)

    for _, r in score_df.sort_values("n_obs", ascending=False).iterrows():
        if r["n_obs"] >= 5 and ok(r["date"]):
            selected.add(r["date"])
            used.add(r["date"])
        if len([d for d in selected if score_df[score_df["date"] == d]["n_obs"].iloc[0] >= 5]) >= 6:
            break
    for _, r in score_df[score_df["n_obs"] <= 1].iterrows():
        if ok(r["date"]):
            selected.add(r["date"])
            used.add(r["date"])
        if len([d for d in selected if score_df[score_df["date"] == d]["n_obs"].iloc[0] <= 1]) >= 3:
            break

    dates = sorted(selected)
    log.info("Selected %d dates: %s", len(dates), dates)

    auto_dir = OUT / "auto_labels"
    for date_str in dates:
        log.info("  Date: %s", date_str)
        ds, n = smart_auto_label(ds, date_str, auto_dir, tau=TAU, hrrr_ds=hrrr_ds,
                                glacier_mask=glacier_mask)
        row = score_df[score_df["date"] == date_str].iloc[0]
        log.info("    %d polygons, %d nearby obs", n, row["n_obs"])

    # Free the big dataset
    del ds
    if hrrr_ds is not None:
        hrrr_ds.close()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Extract patches
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 2: Extract patches ===")

    # Human patches WITHOUT hrrr (3ch baseline)
    human_3ch_dir = OUT / "patches" / "human_3ch"
    log.info("Extracting human patches (3ch, no HRRR)...")
    extract_patches(NC, HUMAN_LABELS, human_3ch_dir, "2026-02-14", hrrr_path=None)

    # Human patches WITH hrrr (4ch)
    human_4ch_dir = OUT / "patches" / "human_4ch"
    log.info("Extracting human patches (4ch, with HRRR)...")
    extract_patches(NC, HUMAN_LABELS, human_4ch_dir, "2026-02-14", hrrr_path=HRRR)

    # Auto patches WITH hrrr (4ch, melt-filtered auto-labels)
    auto_4ch_dir = OUT / "patches" / "auto_4ch"
    log.info("Extracting auto patches (4ch, with HRRR)...")
    for gpkg_file in sorted(auto_dir.glob("avalanche_labels_*.gpkg")):
        date = gpkg_file.stem.replace("avalanche_labels_", "")
        extract_patches(NC, auto_dir, auto_4ch_dir, date, hrrr_path=HRRR)

    # Assign confidence
    log.info("Assigning confidence...")
    for pdir in [human_3ch_dir, human_4ch_dir, auto_4ch_dir]:
        if pdir.exists():
            subprocess.run([sys.executable, "scripts/debris_pixel_classifier/v2/assign_confidence.py",
                            "--patches-dir", str(pdir)], check=False)

    hp3, hn3 = count_patches(human_3ch_dir)
    hp4, hn4 = count_patches(human_4ch_dir)
    ap4, an4 = count_patches(auto_4ch_dir)
    log.info("Human 3ch:  %d pos, %d neg", hp3, hn3)
    log.info("Human 4ch:  %d pos, %d neg", hp4, hn4)
    log.info("Auto 4ch:   %d pos, %d neg", ap4, an4)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Training
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 3: Training ===")

    # Model A: human_only_3ch (baseline — same as before, no HRRR)
    w_human_3ch = weights_dir / "human_only_3ch.pt"
    log.info("Training human_only_3ch (baseline)...")
    train(human_3ch_dir, w_human_3ch, epochs=50, lr=1e-3, sar_ch=3)

    # Model B: human_only_4ch (human labels with melt channel + filtered d_empirical)
    w_human_4ch = weights_dir / "human_only_4ch.pt"
    log.info("Training human_only_4ch (with HRRR melt)...")
    train(human_4ch_dir, w_human_4ch, epochs=50, lr=1e-3, sar_ch=4)

    # Model C: auto pretrain → human finetune (both 4ch with HRRR)
    w_auto_pretrain = weights_dir / "auto_melt_pretrain.pt"
    w_auto_ft = weights_dir / "auto_melt_ft.pt"
    log.info("Training auto_melt pretrain (4ch)...")
    train(auto_4ch_dir, w_auto_pretrain, epochs=50, lr=1e-3, sar_ch=4)
    log.info("Training auto_melt finetune on human (4ch)...")
    train(human_4ch_dir, w_auto_ft, epochs=20, lr=1e-4, resume=w_auto_pretrain, sar_ch=4)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Inference
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 4: Season inference ===")

    model_variants = [
        ("human_3ch", w_human_3ch, 3, None),
        ("human_4ch", w_human_4ch, 4, HRRR),
        ("auto_melt_ft", w_auto_ft, 4, HRRR),
    ]

    for name, weights, sar_ch, hrrr_path in model_variants:
        if not weights.exists():
            log.warning("No weights for %s", name)
            continue
        inf_dir = OUT / f"inference_{name}"
        log.info("Inference: %s (sar_ch=%d, hrrr=%s)", name, sar_ch, hrrr_path is not None)
        run_inference(NC, weights, inf_dir, sar_ch=sar_ch, hrrr_path=hrrr_path)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Validation
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 5: Validation ===")

    from scripts.debris_pixel_classifier.v2.run_cnfaic_experiment import (
        compare_vs_obs, compute_path_f1, print_f1_report, print_dsize_report,
        parse_akrr_obs, merge_obs_sources, temporal_train_val_split,
    )
    from sarvalanche.io.dataset import load_netcdf_to_dataset

    ds = load_netcdf_to_dataset(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    scene_box = box(*ds.rio.bounds())

    # Build val set
    akdot_gdf = gpd.GeoDataFrame(
        obs_2526, geometry=gpd.points_from_xy(obs_2526.longitude, obs_2526.latitude), crs="EPSG:4326",
    )
    akdot_gdf = akdot_gdf[akdot_gdf.within(scene_box)]
    akrr_gdf = parse_akrr_obs()
    all_obs = merge_obs_sources(akdot_gdf, akrr_gdf)
    _, val_obs, split_date = temporal_train_val_split(all_obs)

    akdot_paths = gpd.read_file(str(AKDOT_PATHS))
    paths_in_scene = akdot_paths[akdot_paths.intersects(scene_box)].copy()
    if AKRR_PATHS.exists():
        akrr_paths = gpd.read_file(str(AKRR_PATHS))
        akrr_in = akrr_paths[akrr_paths.intersects(scene_box)]
        existing = set(n.lower() for n in paths_in_scene["name"].values)
        new_akrr = akrr_in[~akrr_in["name"].str.lower().isin(existing)]
        if len(new_akrr) > 0:
            paths_in_scene = pd.concat([paths_in_scene, new_akrr[["name", "geometry"]]], ignore_index=True)
            paths_in_scene = gpd.GeoDataFrame(paths_in_scene, crs=akdot_paths.crs)

    x, y = ds.x.values, ds.y.values
    H, W = len(y), len(x)
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx / 2, float(y.min()) - dy / 2,
        float(x.max()) + dx / 2, float(y.max()) + dy / 2, W, H,
    )
    ds.close()

    log.info("Val obs: %d, Split: %s", len(val_obs), str(split_date)[:10])

    model_names = [name for name, _, _, _ in model_variants]
    all_results = {}

    for name in model_names:
        prob_nc = OUT / f"inference_{name}" / "season_v2_debris_probabilities.nc"
        if not prob_nc.exists():
            log.warning("No inference for %s", name)
            continue
        obs_df = compare_vs_obs(prob_nc, name, paths_in_scene, val_obs,
                                "EPSG:4326", x, y, H, W, transform)
        all_results[name] = obs_df
        if len(obs_df) > 0:
            n20 = (obs_df[f"{name}_n20"] >= 1).sum()
            n50 = (obs_df[f"{name}_n50"] >= 1).sum()
            n = len(obs_df)
            print(f"\n{name}: {n20}/{n} detected (>0.2), {n50}/{n} detected (>0.5)")
            print_dsize_report(obs_df, name)
        f1_df = compute_path_f1(prob_nc, paths_in_scene, val_obs, x, y, H, W, transform)
        print_f1_report(f1_df, name)
        f1_df.to_csv(OUT / f"path_f1_{name}.csv", index=False)

    # Summary
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print(f"OVERALL COMPARISON (held-out val obs after {str(split_date)[:10]})")
        print(f"{'='*80}")
        for name in model_names:
            if name not in all_results:
                continue
            obs_df = all_results[name]
            if len(obs_df) == 0:
                continue
            n20 = (obs_df[f"{name}_n20"] >= 1).sum()
            n50 = (obs_df[f"{name}_n50"] >= 1).sum()
            n = len(obs_df)
            print(f"  {name:20s}  detected(>0.2)={n20}/{n}  detected(>0.5)={n50}/{n}")

    log.info("Results saved to %s", OUT)


if __name__ == "__main__":
    main()
