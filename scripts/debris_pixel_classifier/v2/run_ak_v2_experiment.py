"""AK v2 experiment: manual thresholds + melt filtering + new human labels.

Uses:
  - Manual d_empirical thresholds on melt-filtered signal for auto-labeling
  - Glacier mask, size/runout filters
  - HRRR melt weight as 4th SAR channel + filtered/residual static channels
  - New human labels: Dec 15 (216 polys) + Feb 14 (673 polys)
  - AK data only

Models:
  - human_only_4ch: human labels only (Dec 15 + Feb 14), 4ch SAR, 13 static
  - auto_ft_4ch: auto pretrain → human finetune, all with melt features
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
AKRR_PATHS = Path("local/cnfaic/reported/akrr/akrr_avalanche_paths.gpkg")
HRRR = Path("local/cnfaic/hrrr_temperature_2526.nc")
THRESHOLDS = Path("local/cnfaic/melt_experiment/manual_thresholds.csv")
OUT = Path("local/cnfaic/ak_v2_experiment")
TAU = 6

# Human label dates and their file formats
HUMAN_DATES = {
    "2025-12-15": HUMAN_LABELS / "avalanche_labels_2025-12-15.gpkg",
    "2026-02-14": HUMAN_LABELS / "avalanche_labels_2026-02-14.gpkg.shp",
}


def extract_patches(nc, labels_path, out_dir, date, hrrr_path=None):
    """Extract patches from a label file (gpkg or shp)."""
    if not labels_path.exists():
        log.warning("Labels not found: %s", labels_path)
        return
    cmd = [
        sys.executable, "scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py",
        "--nc", str(nc), "--polygons", str(labels_path),
        "--date", date, "--tau", str(TAU),
        "--out-dir", str(out_dir / date),
        "--stride", "64", "--neg-ratio", "3.0",
        "--pairs", "--max-pairs", "4",
    ]
    if hrrr_path and Path(hrrr_path).exists():
        cmd += ["--hrrr", str(hrrr_path)]
    subprocess.run(cmd, check=False)


def train(data_dirs, out_weights, epochs=50, lr=1e-3, resume=None, sar_ch=4):
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
    subprocess.run(cmd, check=False)


def run_inference(nc, weights, out_dir, sar_ch=4, hrrr_path=None):
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
    if not Path(d).exists():
        return 0, 0
    for lf in Path(d).rglob("labels.json"):
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
    # PHASE 1: Auto-labeling with manual thresholds + melt filtering
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 1: Auto-labeling (manual thresholds, melt-filtered) ===")

    from sarvalanche.io.dataset import load_netcdf_to_dataset
    ds = load_netcdf_to_dataset(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    hrrr_ds = xr.open_dataset(HRRR) if HRRR.exists() else None

    from scripts.debris_pixel_classifier.v2.run_cnfaic_experiment import (
        smart_auto_label, _compute_empirical, build_glacier_mask,
    )
    from sarvalanche.ml.v2.patch_extraction import _compute_melt_filtered_d_empirical

    glacier_mask = build_glacier_mask(ds)

    # Load manual thresholds
    thr_df = pd.read_csv(THRESHOLDS, comment="#")
    manual_thresholds = dict(zip(thr_df["date"], thr_df["threshold_db"]))
    log.info("Loaded %d manual thresholds", len(manual_thresholds))

    # Select in-season dates
    times = pd.DatetimeIndex(ds.time.values)
    in_season = times[(times.month >= 11) | (times.month <= 4)]
    all_dates = sorted(set(t.strftime("%Y-%m-%d") for t in in_season))

    auto_dir = OUT / "auto_labels"
    total_polys = 0
    for date_str in all_dates:
        threshold = manual_thresholds.get(date_str)
        if threshold is None:
            log.info("  %s: no manual threshold, skipping", date_str)
            continue

        log.info("  %s (threshold=%.1f dB)...", date_str, threshold)

        # Compute melt-filtered d_empirical
        ds = _compute_empirical(ds, date_str, TAU)
        H, W = ds.sizes["y"], ds.sizes["x"]
        if hrrr_ds is not None:
            d = _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W)
            if d is None:
                d = ds["d_empirical"].values
            else:
                log.info("    Using melt-filtered d_empirical")
        else:
            d = ds["d_empirical"].values

        # Apply manual threshold directly instead of Otsu
        from scipy import ndimage
        from rasterio.features import shapes as rasterio_shapes
        from shapely.geometry import shape

        slope = ds["slope"].values
        valid = (slope >= 0.09) & (slope <= 1.05) & np.isfinite(d)
        if "water_mask" in ds:
            valid = valid & (ds["water_mask"].values == 0)

        candidates = valid & (d >= threshold)

        # FlowPy boost
        if "cell_counts" in ds:
            cc = ds["cell_counts"].values
            boost = valid & (d >= threshold * 0.75) & (cc > 0)
            candidates = candidates | boost

        labeled, n_comp = ndimage.label(candidates)
        if n_comp == 0:
            log.info("    0 polygons")
            continue

        dx = abs(float(ds.x.values[1] - ds.x.values[0]))
        dy = abs(float(ds.y.values[1] - ds.y.values[0]))
        crs = ds.rio.crs
        if crs and crs.is_geographic:
            mid_lat = float(ds.y.values.mean())
            px_area = dx * 111320 * np.cos(np.radians(mid_lat)) * dy * 110540
        else:
            px_area = dx * dy

        x_vals, y_vals = ds.x.values, ds.y.values
        transform = from_bounds(
            float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
            float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2,
            len(x_vals), len(y_vals),
        )

        sizes = ndimage.sum(candidates, labeled, range(1, n_comp + 1))
        cc_vals = ds["cell_counts"].values if "cell_counts" in ds else None

        polygons = []
        n_rej_size = n_rej_glacier = n_rej_runout = 0
        for comp_id in range(1, n_comp + 1):
            size_px = sizes[comp_id - 1]
            area_m2 = size_px * px_area
            if area_m2 < 5_000 or area_m2 > 750_000 or size_px < 10:
                n_rej_size += 1
                continue
            comp_mask = labeled == comp_id
            if glacier_mask is not None:
                if glacier_mask[comp_mask].sum() / max(size_px, 1) > 0.20:
                    n_rej_glacier += 1
                    continue
            if cc_vals is not None:
                if float(np.nanmean(cc_vals[comp_mask])) < 50:
                    n_rej_runout += 1
                    continue
            mean_d = float(np.nanmean(d[comp_mask]))
            mean_slope = float(np.nanmean(slope[comp_mask]))
            comp_bin = (labeled == comp_id).astype(np.uint8)
            for geom, val in rasterio_shapes(comp_bin, transform=transform):
                if val == 1:
                    poly = shape(geom)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append({
                            "geometry": poly, "mean_d": round(mean_d, 2),
                            "area_m2": round(area_m2, 0), "source": "auto",
                            "mean_slope_rad": round(mean_slope, 3),
                        })

        log.info("    %d comp, %d rej(size), %d rej(glacier), %d rej(runout), %d kept",
                 n_comp, n_rej_size, n_rej_glacier, n_rej_runout, len(polygons))

        if len(polygons) > 500:
            polygons.sort(key=lambda p: p["mean_d"], reverse=True)
            polygons = polygons[:500]

        if polygons:
            gdf = gpd.GeoDataFrame(polygons, crs=ds.rio.crs)
            auto_dir.mkdir(parents=True, exist_ok=True)
            gdf.to_file(auto_dir / f"avalanche_labels_{date_str}.gpkg", driver="GPKG")
            total_polys += len(gdf)
            log.info("    Saved %d polygons (%.1f km2)",
                     len(gdf), gdf["area_m2"].sum() / 1e6)

    log.info("Total auto polygons: %d", total_polys)
    del ds
    if hrrr_ds is not None:
        hrrr_ds.close()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Extract patches
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 2: Extract patches ===")

    human_dir = OUT / "patches" / "human"
    auto_patch_dir = OUT / "patches" / "auto"

    # Human labels (both dates, with HRRR)
    for date_str, label_path in HUMAN_DATES.items():
        log.info("Human patches: %s", date_str)
        extract_patches(NC, label_path, human_dir, date_str, hrrr_path=HRRR)

    # Auto labels (with HRRR)
    for gpkg_file in sorted(auto_dir.glob("avalanche_labels_*.gpkg")):
        date = gpkg_file.stem.replace("avalanche_labels_", "")
        log.info("Auto patches: %s", date)
        extract_patches(NC, gpkg_file, auto_patch_dir, date, hrrr_path=HRRR)

    # Assign confidence
    log.info("Assigning confidence...")
    for pdir in [human_dir, auto_patch_dir]:
        if pdir.exists():
            subprocess.run([sys.executable, "scripts/debris_pixel_classifier/v2/assign_confidence.py",
                            "--patches-dir", str(pdir)], check=False)

    hp, hn = count_patches(human_dir)
    ap, an = count_patches(auto_patch_dir)
    log.info("Human patches: %d pos, %d neg", hp, hn)
    log.info("Auto patches:  %d pos, %d neg", ap, an)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Training
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 3: Training ===")

    # Model A: human_only (4ch, 13 static)
    w_human = weights_dir / "human_only_4ch.pt"
    log.info("Training human_only_4ch...")
    train(human_dir, w_human, epochs=50, lr=1e-3, sar_ch=4)

    # Model B: auto pretrain → human finetune (4ch)
    w_auto_pt = weights_dir / "auto_pretrain_4ch.pt"
    w_auto_ft = weights_dir / "auto_ft_4ch.pt"
    log.info("Training auto pretrain (4ch)...")
    train(auto_patch_dir, w_auto_pt, epochs=50, lr=1e-3, sar_ch=4)
    log.info("Training auto → human finetune (4ch)...")
    train(human_dir, w_auto_ft, epochs=30, lr=1e-4, resume=w_auto_pt, sar_ch=4)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Inference
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 4: Inference ===")

    models = [
        ("human_only_4ch", w_human),
        ("auto_ft_4ch", w_auto_ft),
    ]
    for name, weights in models:
        if not weights.exists():
            log.warning("No weights: %s", name)
            continue
        log.info("Inference: %s", name)
        run_inference(NC, weights, OUT / f"inference_{name}", sar_ch=4, hrrr_path=HRRR)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Validation
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 5: Validation ===")

    from scripts.debris_pixel_classifier.v2.run_cnfaic_experiment import (
        compare_vs_obs, compute_path_f1, print_f1_report, print_dsize_report,
        parse_akrr_obs, merge_obs_sources, temporal_train_val_split,
    )
    from sarvalanche.io.dataset import load_netcdf_to_dataset as _load

    ds = _load(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    scene_box = box(*ds.rio.bounds())

    obs_all = pd.read_csv(str(AKDOT_OBS))
    obs_all["avalanche_date"] = pd.to_datetime(obs_all["avalanche_date"])
    obs_2526 = obs_all[(obs_all["avalanche_date"] >= "2025-11-01") & (obs_all["avalanche_date"] <= "2026-04-30")]
    obs_2526 = obs_2526[obs_2526["latitude"].notna()].copy()
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

    model_names = [n for n, _ in models]
    for name in model_names:
        prob_nc = OUT / f"inference_{name}" / "season_v2_debris_probabilities.nc"
        if not prob_nc.exists():
            continue
        obs_df = compare_vs_obs(prob_nc, name, paths_in_scene, val_obs,
                                "EPSG:4326", x, y, H, W, transform)
        if len(obs_df) > 0:
            n20 = (obs_df[f"{name}_n20"] >= 1).sum()
            n50 = (obs_df[f"{name}_n50"] >= 1).sum()
            n = len(obs_df)
            print(f"\n{name}: {n20}/{n} detected (>0.2), {n50}/{n} detected (>0.5)")
            print_dsize_report(obs_df, name)
        f1_df = compute_path_f1(prob_nc, paths_in_scene, val_obs, x, y, H, W, transform)
        print_f1_report(f1_df, name)
        f1_df.to_csv(OUT / f"path_f1_{name}.csv", index=False)

    log.info("Results saved to %s", OUT)


if __name__ == "__main__":
    main()
