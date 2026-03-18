"""CNFAIC experiment: auto-label → pretrain → human-finetune → validate against AKDOT obs.

Uses smart multi-feature thresholding for auto-labeling:
  - Elevation-banded Otsu thresholds on d_empirical
  - Cross-ratio (VH-VV) wet-snow suppression
  - FlowPy runout boosting
  - Quiet-period diagnostics
Validates with F1 and false positive rates at the path level.
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

RGI_SHP = Path("local/cnfaic/rgi/RGI2000-v7.0-G-01_alaska.shp")
NC = Path("local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc")
NC_2425 = Path("local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2024-2025_Turnagain_Pass_and_Girdwood.nc")
HUMAN_LABELS = Path("local/cnfaic/debris_shapes")
AKDOT_OBS = Path("local/cnfaic/reported/akdot/akdot_avalanche_observations.csv")
AKDOT_PATHS = Path("local/cnfaic/reported/akdot/avy_path_frequency.gpkg")
AKRR_OBS = Path("local/cnfaic/reported/akrr/akrr_avalanche_obs.csv")
AKRR_PATHS = Path("local/cnfaic/reported/akrr/akrr_avalanche_paths.gpkg")
SNFAC_PATCHES = Path("local/issw/v2_patches_pairs")
OUT = Path("local/cnfaic/cnn_experiment")
TAU = 6
TRAIN_FRAC = 0.6  # temporal train/val split

def build_glacier_mask(ds):
    """Rasterize RGI glacier polygons to the SAR grid. Returns (H, W) bool array."""
    if not RGI_SHP.exists():
        log.warning("RGI shapefile not found at %s, skipping glacier mask", RGI_SHP)
        return None

    rgi = gpd.read_file(str(RGI_SHP))
    scene_box = box(*ds.rio.bounds())
    rgi_in = rgi[rgi.intersects(scene_box)]
    if len(rgi_in) == 0:
        log.info("No glaciers in scene")
        return np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)

    # Rasterize to SAR grid
    H, W = ds.sizes["y"], ds.sizes["x"]
    x_vals, y_vals = ds.x.values, ds.y.values
    dx = abs(float(x_vals[1] - x_vals[0]))
    dy = abs(float(y_vals[1] - y_vals[0]))
    transform = from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2, W, H,
    )

    glacier_mask = ~rasterio.features.geometry_mask(
        rgi_in.geometry, out_shape=(H, W), transform=transform, all_touched=True,
    )
    log.info("Glacier mask: %d glaciers, %d pixels (%.1f%% of grid)",
             len(rgi_in), glacier_mask.sum(), 100 * glacier_mask.mean())
    return glacier_mask


# ── Slope bounds (radians) ───────────────────────────────────────────
SLOPE_MIN_RAD = 0.09   # ~5°
SLOPE_MAX_RAD = 1.05   # ~60°


# ── Phase 1: Smart auto-labeling ─────────────────────────────────────

def _compute_empirical(ds, date_str, tau):
    """Compute d_empirical and p_empirical for a date, cleaning stale vars."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(pd.Timestamp(date_str))

    stale = [v for v in ds.data_vars if re.match(r"^[pdm]_\d+_V[VH]_empirical$", v)
             or v in {"p_empirical", "d_empirical", "w_temporal"}]
    if stale:
        ds = ds.drop_vars(stale)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_emp, d_emp = calculate_empirical_backscatter_probability(
        ds, ref_ts, use_agreement_boosting=True,
        agreement_strength=0.8, min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_emp
    ds["d_empirical"] = d_emp
    return ds


def _compute_cross_ratio_change(ds, date_str, tau):
    """Compute cross-ratio change (VH-VV in dB) around the reference date.

    Negative values indicate wet-snow onset (VH drops more than VV).
    Returns a 2-D array (y, x) or None if both pols aren't available.
    """
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(pd.Timestamp(date_str))
    w_t = get_temporal_weights(ds["time"], ref_ts, tau_days=tau).values

    # Need both VV and VH
    has_vv = "VV" in ds.data_vars
    has_vh = "VH" in ds.data_vars
    if not (has_vv and has_vh):
        return None

    vv = ds["VV"].values  # (time, y, x), linear
    vh = ds["VH"].values

    # Cross ratio = VH/VV in linear, or VH_dB - VV_dB
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where((vv > 0) & (vh > 0), 10 * np.log10(vh / vv), np.nan)

    # Weighted mean before and after reference date
    times = ds["time"].values
    before = times < ref_ts
    after = times >= ref_ts

    if before.sum() == 0 or after.sum() == 0:
        return None

    w_before = w_t.copy()
    w_before[~before] = 0
    w_after = w_t.copy()
    w_after[~after] = 0

    s_b = w_before.sum()
    s_a = w_after.sum()
    if s_b == 0 or s_a == 0:
        return None

    # Weighted mean CR before and after
    cr_before = np.nansum(cr * w_before[:, None, None], axis=0) / s_b
    cr_after = np.nansum(cr * w_after[:, None, None], axis=0) / s_a

    # Change: positive = CR increased (drier), negative = CR dropped (wetter)
    d_cr = cr_after - cr_before
    return d_cr


def _elevation_banded_otsu(d, dem, valid, band_size=200, min_px_per_band=500):
    """Compute per-pixel threshold using Otsu within elevation bands.

    Falls back to 98th percentile if Otsu fails (e.g., unimodal distribution).
    Returns a 2-D threshold array matching d's shape.
    """
    from skimage.filters import threshold_otsu

    threshold_map = np.full_like(d, np.nan)
    dem_flat = dem.copy()
    dem_flat[~valid] = np.nan

    lo = np.nanmin(dem_flat)
    hi = np.nanmax(dem_flat)
    edges = np.arange(lo, hi + band_size, band_size)

    band_thresholds = []
    for i in range(len(edges) - 1):
        band_mask = valid & (dem >= edges[i]) & (dem < edges[i + 1])
        n = band_mask.sum()
        if n < min_px_per_band:
            band_thresholds.append((edges[i], edges[i + 1], np.nan, n))
            continue

        d_band = d[band_mask]
        try:
            thresh = threshold_otsu(d_band)
        except Exception:
            thresh = np.percentile(d_band, 98)

        # Floor at 2.0 dB — below this we can't reliably distinguish debris
        thresh = max(thresh, 2.0)
        threshold_map[band_mask] = thresh
        band_thresholds.append((edges[i], edges[i + 1], thresh, n))

    # Fill bands that had too few pixels: interpolate from neighbors
    valid_bands = [(lo_e, hi_e, t, n) for lo_e, hi_e, t, n in band_thresholds if not np.isnan(t)]
    if not valid_bands:
        # All bands failed — fall back to global 98th percentile
        global_thresh = max(float(np.percentile(d[valid], 98)), 2.0)
        threshold_map[valid] = global_thresh
        return threshold_map

    # For missing bands, use nearest valid band's threshold
    for lo_e, hi_e, t, n in band_thresholds:
        if np.isnan(t):
            band_mask = valid & (dem >= lo_e) & (dem < hi_e)
            if band_mask.sum() == 0:
                continue
            mid = (lo_e + hi_e) / 2
            nearest = min(valid_bands, key=lambda b: abs((b[0] + b[1]) / 2 - mid))
            threshold_map[band_mask] = nearest[2]

    return threshold_map


def smart_auto_label(ds, date_str, out_dir, tau=TAU, hrrr_ds=None, glacier_mask=None):
    """Auto-label with elevation-banded Otsu + cross-ratio wet-snow suppression.

    Improvements over simple MAD threshold:
      1. Elevation-banded Otsu: adapts to elevation-dependent melt/brightness
      2. Cross-ratio penalty: suppresses wet-snow false positives
      3. Multi-feature composite: slope feasibility + FlowPy as positive signal
      4. Quiet-period diagnostics: logs polygon counts for sanity checking
      5. HRRR melt filtering: uses melt-filtered d_empirical when hrrr_ds provided
    """
    from scipy import ndimage
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape

    # Compute d_empirical
    try:
        ds = _compute_empirical(ds, date_str, tau)
    except Exception as e:
        log.warning("  Empirical failed for %s: %s", date_str, e)
        return ds, 0

    # Use melt-filtered d_empirical if HRRR available
    if hrrr_ds is not None:
        from sarvalanche.ml.v2.patch_extraction import _compute_melt_filtered_d_empirical
        H, W = ds.sizes["y"], ds.sizes["x"]
        d_filtered = _compute_melt_filtered_d_empirical(ds, hrrr_ds, H, W)
        if d_filtered is not None:
            log.info("    Using melt-filtered d_empirical")
            d = d_filtered
        else:
            d = ds["d_empirical"].values
    else:
        d = ds["d_empirical"].values
    slope = ds["slope"].values
    has_water = "water_mask" in ds.data_vars
    has_dem = "dem" in ds.data_vars
    has_cc = "cell_counts" in ds.data_vars

    # Valid terrain mask
    valid = (slope >= SLOPE_MIN_RAD) & (slope <= SLOPE_MAX_RAD) & np.isfinite(d)
    if has_water:
        valid = valid & (ds["water_mask"].values == 0)

    if valid.sum() < 100:
        log.warning("  Too few valid pixels for %s", date_str)
        return ds, 0

    # ── 0. Scene-wide melt detection ─────────────────────────────────
    # If a large fraction of valid terrain has elevated d_empirical,
    # the scene is likely dominated by a melt or rain-on-snow event.
    # In that case, raise the threshold floor to suppress false positives.
    d_valid = d[valid]
    bright_frac_2db = float((d_valid >= 2.0).sum()) / len(d_valid)
    bright_frac_3db = float((d_valid >= 3.0).sum()) / len(d_valid)
    melt_flag = False
    melt_threshold_boost = 1.0

    if bright_frac_3db > 0.20:
        # >20% of terrain above 3 dB = very likely scene-wide melt
        melt_flag = True
        melt_threshold_boost = 1.5
        log.warning("    MELT FLAG: %.1f%% of valid terrain > 3 dB — "
                     "raising thresholds by 1.5x", bright_frac_3db * 100)
    elif bright_frac_2db > 0.30:
        # >30% above 2 dB = moderate melt concern
        melt_flag = True
        melt_threshold_boost = 1.25
        log.warning("    MELT FLAG: %.1f%% of valid terrain > 2 dB — "
                     "raising thresholds by 1.25x", bright_frac_2db * 100)
    else:
        log.info("    Scene brightness: %.1f%% > 2 dB, %.1f%% > 3 dB (OK)",
                 bright_frac_2db * 100, bright_frac_3db * 100)

    # ── 1. Elevation-banded Otsu threshold ────────────────────────────
    if has_dem:
        dem = ds["dem"].values
        threshold_map = _elevation_banded_otsu(d, dem, valid, band_size=200)
        # Log per-band thresholds
        lo, hi = np.nanmin(dem[valid]), np.nanmax(dem[valid])
        edges = np.arange(lo, hi + 200, 200)
        for i in range(len(edges) - 1):
            band = valid & (dem >= edges[i]) & (dem < edges[i + 1])
            if band.sum() > 0:
                t = np.nanmedian(threshold_map[band])
                log.info("    Elev %4.0f–%4.0fm: threshold=%.2f dB (%d px)",
                         edges[i], edges[i + 1], t, band.sum())
    else:
        # No DEM: fall back to global Otsu
        from skimage.filters import threshold_otsu
        try:
            global_thresh = max(threshold_otsu(d[valid]), 2.0)
        except Exception:
            global_thresh = max(float(np.percentile(d[valid], 98)), 2.0)
        threshold_map = np.where(valid, global_thresh, np.nan)
        log.info("    Global Otsu threshold: %.2f dB (no DEM)", global_thresh)

    # ── 2. Cross-ratio wet-snow suppression ───────────────────────────
    d_cr = _compute_cross_ratio_change(ds, date_str, tau)
    wet_snow_penalty = np.ones_like(d)
    if d_cr is not None:
        # Negative d_cr = wet-snow onset → raise effective threshold
        # Scale: d_cr < -2 dB → penalty 2.0 (doubles threshold)
        #        d_cr ~ 0    → penalty 1.0 (no change)
        penalty = np.clip(1.0 - d_cr / 4.0, 1.0, 2.0)  # d_cr=-4 → 2.0, d_cr=0 → 1.0
        wet_snow_penalty = np.where(np.isfinite(d_cr), penalty, 1.0)
        n_penalized = (wet_snow_penalty > 1.1).sum()
        log.info("    Cross-ratio: %d pixels penalized (%.1f%% of valid)",
                 n_penalized, 100 * n_penalized / valid.sum())

    # ── 3. Apply thresholds: d >= threshold * wet_penalty * melt_boost ─
    effective_threshold = threshold_map * wet_snow_penalty * melt_threshold_boost
    candidates = valid & np.isfinite(effective_threshold) & (d >= effective_threshold)

    # ── 4. FlowPy boost (lower threshold in modeled runout zones) ─────
    if has_cc:
        cc = ds["cell_counts"].values
        boost = valid & np.isfinite(effective_threshold) & (d >= effective_threshold * 0.75) & (cc > 0)
        n_boosted = boost.sum() - candidates.sum()
        candidates = candidates | boost
        if n_boosted > 0:
            log.info("    FlowPy boost added %d candidate pixels", max(n_boosted, 0))

    # ── 5. Connected components + polygon extraction ──────────────────
    labeled, n_comp = ndimage.label(candidates)
    if n_comp == 0:
        log.info("  %s: 0 polygons (no candidates passed threshold)", date_str)
        return ds, 0

    # Pixel area
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        mid_lat = float(ds.y.values.mean())
        px_area = dx * 111320 * np.cos(np.radians(mid_lat)) * dy * 110540
    else:
        px_area = dx * dy

    x_vals, y_vals = ds.x.values, ds.y.values
    H, W = len(y_vals), len(x_vals)
    transform = from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2, W, H,
    )

    sizes = ndimage.sum(candidates, labeled, range(1, n_comp + 1))
    # Cell counts for runout overlap check
    has_cc_data = has_cc and "cell_counts" in ds.data_vars
    cc_vals = ds["cell_counts"].values if has_cc_data else None

    # Water mask for component rejection
    water_vals = ds["water_mask"].values if has_water else None

    polygons = []
    n_rejected_size = 0
    n_rejected_runout = 0
    n_rejected_glacier = 0
    n_rejected_water = 0
    for comp_id in range(1, n_comp + 1):
        size_px = sizes[comp_id - 1]
        area_m2 = size_px * px_area
        # Size filter: based on human label distribution
        # Human labels: 5th pct = 11,000 m², 99th = 681,000 m²
        if area_m2 < 5_000 or area_m2 > 750_000 or size_px < 10:
            n_rejected_size += 1
            continue
        comp_mask = labeled == comp_id
        # Reject components with >20% glacier overlap
        if glacier_mask is not None:
            glacier_frac = glacier_mask[comp_mask].sum() / max(size_px, 1)
            if glacier_frac > 0.20:
                n_rejected_glacier += 1
                continue
        # Reject components with >10% water overlap
        if water_vals is not None:
            water_frac = (water_vals[comp_mask] == 1).sum() / max(size_px, 1)
            if water_frac > 0.10:
                n_rejected_water += 1
                continue
        # Require mean cell_count ≥ 50 in the component
        # Real avalanche debris concentrates in modeled runout fans
        # (mean cc ≥ 50 keeps ~68% of candidates, removes detections on
        # terrain where FlowPy doesn't predict significant deposition)
        if cc_vals is not None:
            mean_cc = float(np.nanmean(cc_vals[comp_mask]))
            if mean_cc < 50:
                n_rejected_runout += 1
                continue
        mean_d = float(np.nanmean(d[comp_mask]))
        mean_slope = float(np.nanmean(slope[comp_mask]))
        # Per-polygon cross-ratio info
        mean_cr = float(np.nanmean(d_cr[comp_mask])) if d_cr is not None else None
        comp_bin = (labeled == comp_id).astype(np.uint8)
        for geom, val in rasterio_shapes(comp_bin, transform=transform):
            if val == 1:
                poly = shape(geom)
                if poly.is_valid and not poly.is_empty:
                    rec = {"geometry": poly, "mean_d": round(mean_d, 2),
                           "area_m2": round(area_m2, 0), "source": "auto",
                           "mean_slope_rad": round(mean_slope, 3)}
                    if mean_cr is not None:
                        rec["mean_d_cr"] = round(mean_cr, 2)
                    polygons.append(rec)

    log.info("    Components: %d total, %d rejected (size), %d rejected (glacier), "
             "%d rejected (water), %d rejected (runout), %d kept",
             n_comp, n_rejected_size, n_rejected_glacier, n_rejected_water,
             n_rejected_runout, len(polygons))

    # Cap: tighter on melt-flagged dates (too many detections = melt noise)
    max_polys = 200 if melt_flag else 500
    if len(polygons) > max_polys:
        polygons.sort(key=lambda p: p["mean_d"], reverse=True)
        polygons = polygons[:max_polys]
        log.warning("  Capped at %d polygons%s", max_polys,
                     " (melt-flagged)" if melt_flag else "")

    if polygons:
        gdf = gpd.GeoDataFrame(polygons, crs=ds.rio.crs)
        out_path = out_dir / f"avalanche_labels_{date_str}.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GPKG")
        total_area = gdf["area_m2"].sum()
        log.info("  %s: %d polygons, total %.0f m² (mean_d range: %.1f–%.1f dB)",
                 date_str, len(gdf), total_area,
                 gdf["mean_d"].min(), gdf["mean_d"].max())
    else:
        log.info("  %s: 0 polygons after size filtering", date_str)

    return ds, len(polygons)


# ── Phase 1b: Observation-guided labeling ─────────────────────────────

def obs_guided_label(ds, obs_gdf, akdot_paths, out_dir, tau=TAU,
                     d_threshold=1.5, min_component_px=3):
    """Generate training labels from AKDOT observations + avalanche paths.

    For each observation we KNOW an avalanche happened, so we can use a much
    lower d_empirical threshold (1.5 dB vs ~3+ for blind auto-labeling) to
    find the debris pixels within the confirmed path polygon.

    Groups observations by closest SAR date, computes d_empirical once per
    SAR date, then extracts debris polygons from each confirmed path.

    Returns dict of {date_str: n_polygons}.
    """
    from scipy import ndimage
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape

    # Grid info
    x_vals, y_vals = ds.x.values, ds.y.values
    H, W = len(y_vals), len(x_vals)
    dx = abs(float(x_vals[1] - x_vals[0]))
    dy = abs(float(y_vals[1] - y_vals[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        mid_lat = float(y_vals.mean())
        px_area = dx * 111320 * np.cos(np.radians(mid_lat)) * dy * 110540
    else:
        px_area = dx * dy
    grid_transform = from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2, W, H,
    )

    sar_times = pd.DatetimeIndex(ds.time.values)
    slope = ds["slope"].values

    # Match each obs to closest SAR date (within tau days)
    obs_by_sar = {}  # sar_date_str -> list of (obs_row, path_geom)
    for _, obs_row in obs_gdf.iterrows():
        obs_date = obs_row["avalanche_date"]
        time_diffs = np.abs(sar_times - obs_date)
        ci = time_diffs.argmin()
        if time_diffs[ci].days > tau:
            continue

        # Find containing path
        obs_point = obs_row.geometry
        containing = akdot_paths[akdot_paths.contains(obs_point)]
        if len(containing) == 0:
            buf = akdot_paths.to_crs("EPSG:3338").copy()
            buf["geometry"] = buf.geometry.buffer(150)
            buf = buf.to_crs("EPSG:4326")
            containing = buf[buf.contains(obs_point)]
        if len(containing) == 0:
            continue

        sar_date_str = str(sar_times[ci])[:10]
        path_geom = containing.iloc[0].geometry
        path_name = containing.iloc[0]["name"]
        obs_by_sar.setdefault(sar_date_str, []).append(
            (obs_row, path_geom, path_name)
        )

    log.info("  Matched %d obs to %d SAR dates",
             sum(len(v) for v in obs_by_sar.values()), len(obs_by_sar))

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    for sar_date_str, obs_list in sorted(obs_by_sar.items()):
        # Compute d_empirical for this SAR date
        try:
            ds = _compute_empirical(ds, sar_date_str, tau)
        except Exception as e:
            log.warning("  Empirical failed for %s: %s", sar_date_str, e)
            continue

        d = ds["d_empirical"].values
        polygons = []
        paths_used = set()

        for obs_row, path_geom, path_name in obs_list:
            if path_name in paths_used:
                continue  # avoid duplicate polygons for same path
            paths_used.add(path_name)

            # Rasterize path polygon
            try:
                path_mask = ~rasterio.features.geometry_mask(
                    [path_geom], out_shape=(H, W), transform=grid_transform,
                    all_touched=True,
                )
            except Exception:
                continue
            if path_mask.sum() == 0:
                continue

            # Within the confirmed path, use LOW threshold on d_empirical
            # We know debris is here, so 1.5 dB is enough
            valid_in_path = (
                path_mask
                & (slope >= SLOPE_MIN_RAD) & (slope <= SLOPE_MAX_RAD)
                & np.isfinite(d)
                & (d >= d_threshold)
            )

            if valid_in_path.sum() < min_component_px:
                continue

            # Connected components within path
            labeled, n_comp = ndimage.label(valid_in_path)
            if n_comp == 0:
                continue

            sizes = ndimage.sum(valid_in_path, labeled, range(1, n_comp + 1))
            for comp_id in range(1, n_comp + 1):
                size_px = sizes[comp_id - 1]
                if size_px < min_component_px:
                    continue
                area_m2 = size_px * px_area
                if area_m2 < 300 or area_m2 > 2_000_000:
                    continue
                comp_mask = labeled == comp_id
                mean_d = float(np.nanmean(d[comp_mask]))
                comp_bin = (labeled == comp_id).astype(np.uint8)
                for geom, val in rasterio_shapes(comp_bin, transform=grid_transform):
                    if val == 1:
                        poly = shape(geom)
                        if poly.is_valid and not poly.is_empty:
                            dsize = obs_row.get("dsize", "")
                            polygons.append({
                                "geometry": poly,
                                "mean_d": round(mean_d, 2),
                                "area_m2": round(area_m2, 0),
                                "source": "obs_guided",
                                "path_name": path_name,
                                "dsize": str(dsize),
                            })

        if polygons:
            gdf = gpd.GeoDataFrame(polygons, crs=ds.rio.crs)
            out_path = out_dir / f"avalanche_labels_{sar_date_str}.gpkg"
            gdf.to_file(out_path, driver="GPKG")
            log.info("  %s: %d obs-guided polygons from %d paths",
                     sar_date_str, len(gdf), len(paths_used))
        result[sar_date_str] = len(polygons)

    return ds, result


# ── AKRR parsing + train/val split ────────────────────────────────────

EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _parse_excel_or_text_date(val, last_date=None):
    """Parse a date from AKRR CSV — Excel serial number or freeform string."""
    if pd.isna(val) or str(val).strip() == "":
        return last_date
    s = str(val).strip().rstrip("`")
    # Excel serial number (e.g. 45954)
    try:
        num = float(s)
        if num > 40000:
            return EXCEL_EPOCH + pd.Timedelta(days=int(num))
    except ValueError:
        pass
    # Freeform date range like "10/30-31/25" or "12/2-9/2025" or "3/6-8/2026"
    # Take the first date in the range
    m = re.match(r"(\d{1,2})/(\d{1,2})(?:-\d+)?/(\d{2,4})", s)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            pass
    return last_date


def _extract_dsize(classification_str):
    """Extract D-size from AKRR classification like 'SSw-N-2.5' → 'D2.5'."""
    if pd.isna(classification_str):
        return ""
    s = str(classification_str).strip()
    m = re.search(r"(\d+\.?\d*)\s*$", s)
    if m:
        return f"D{m.group(1)}"
    return ""


def _find_classification(row):
    """Find the avalanche classification string, handling shifted columns."""
    pattern = re.compile(r"[A-Z]{1,3}w?-[A-Z]{1,3}-[\d.]+")
    for col in ["CLASSIFICATION", "NOTES"]:
        val = str(row.get(col, ""))
        if pattern.search(val):
            return val
    # Sometimes classification is in the PATH column
    val = str(row.get("PATH", ""))
    if pattern.search(val):
        return val
    return ""


def parse_akrr_obs():
    """Parse messy AKRR CSV into a GeoDataFrame compatible with obs_guided_label.

    Returns GeoDataFrame with columns: avalanche_date, path_name, dsize, geometry
    where geometry is the centroid of the matched AKRR path polygon.
    """
    if not AKRR_OBS.exists() or not AKRR_PATHS.exists():
        log.warning("AKRR files not found, skipping")
        return gpd.GeoDataFrame(columns=["avalanche_date", "path_name", "dsize", "geometry"])

    akrr_paths = gpd.read_file(str(AKRR_PATHS))
    path_names = {n.lower(): n for n in akrr_paths["name"].values}
    path_centroids = {row["name"]: row.geometry.centroid for _, row in akrr_paths.iterrows()}

    raw = pd.read_csv(str(AKRR_OBS))
    # Drop empty rows
    raw = raw.dropna(how="all")

    rows = []
    last_date = None
    for _, r in raw.iterrows():
        date = _parse_excel_or_text_date(r.get("DATE"), last_date)
        if date is not None:
            last_date = date
        if date is None:
            continue

        classification = _find_classification(r)
        dsize = _extract_dsize(classification)

        # PATH column may contain multiple paths: "Dog Leg, SS, Dump"
        path_str = str(r.get("PATH", ""))
        if pd.isna(r.get("PATH")):
            continue
        # Skip rows that are clearly notes, not path names
        if len(path_str) > 80:
            continue

        # Split on comma for multi-path rows
        candidate_names = [p.strip() for p in path_str.split(",")]

        for cname in candidate_names:
            if not cname or len(cname) < 2:
                continue
            # Skip numeric-only entries (times like "1345", "2028", shot numbers)
            try:
                float(cname)
                continue
            except ValueError:
                pass
            # Skip time-like entries
            if cname.startswith("(") or cname.lower() in ("of note", "night"):
                continue

            # Match to AKRR path names
            matched_name = None
            cname_lower = cname.lower()
            # Exact match
            if cname_lower in path_names:
                matched_name = path_names[cname_lower]
            else:
                # Substring match
                for pn_lower, pn in path_names.items():
                    if cname_lower in pn_lower or pn_lower in cname_lower:
                        matched_name = pn
                        break

            if matched_name is None:
                continue

            rows.append({
                "avalanche_date": date,
                "path_name": matched_name,
                "dsize": dsize,
                "geometry": path_centroids[matched_name],
            })

    if not rows:
        return gpd.GeoDataFrame(columns=["avalanche_date", "path_name", "dsize", "geometry"])

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # Deduplicate same path on same date
    gdf = gdf.drop_duplicates(subset=["avalanche_date", "path_name"])
    log.info("Parsed %d AKRR observations across %d unique dates",
             len(gdf), gdf["avalanche_date"].nunique())
    return gdf


def merge_obs_sources(akdot_gdf, akrr_gdf):
    """Merge AKDOT and AKRR observations into a single GeoDataFrame.

    Normalizes to common schema. AKDOT preferred for duplicates.
    """
    parts = []
    if len(akdot_gdf) > 0:
        akdot = akdot_gdf[["avalanche_date", "geometry"]].copy()
        akdot["dsize"] = akdot_gdf.get("dsize", "")
        akdot["source"] = "akdot"
        # Use objectid if available for dedup
        if "objectid" in akdot_gdf.columns:
            akdot["obs_id"] = akdot_gdf["objectid"]
        else:
            akdot["obs_id"] = range(len(akdot))
        parts.append(akdot)

    if len(akrr_gdf) > 0:
        akrr = akrr_gdf[["avalanche_date", "path_name", "dsize", "geometry"]].copy()
        akrr["source"] = "akrr"
        akrr["obs_id"] = range(10000, 10000 + len(akrr))
        parts.append(akrr)

    if not parts:
        return gpd.GeoDataFrame(columns=["avalanche_date", "dsize", "geometry", "source", "obs_id"])

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    log.info("Merged observations: %d total (%d AKDOT, %d AKRR)",
             len(merged),
             len(merged[merged["source"] == "akdot"]),
             len(merged[merged["source"] == "akrr"]))
    return merged


def temporal_train_val_split(obs_gdf, train_frac=TRAIN_FRAC):
    """Split observations temporally — first train_frac of season for training, rest for validation.

    Returns (train_gdf, val_gdf, split_date).
    """
    dates = sorted(obs_gdf["avalanche_date"].unique())
    if len(dates) == 0:
        empty = obs_gdf.iloc[:0].copy()
        return empty, empty, None

    split_idx = int(len(dates) * train_frac)
    split_date = dates[min(split_idx, len(dates) - 1)]

    train_gdf = obs_gdf[obs_gdf["avalanche_date"] <= split_date].copy()
    val_gdf = obs_gdf[obs_gdf["avalanche_date"] > split_date].copy()

    log.info("Temporal split at %s: %d train obs, %d val obs",
             str(split_date)[:10], len(train_gdf), len(val_gdf))
    return train_gdf, val_gdf, split_date


def print_dsize_report(obs_df, model_name):
    """Print detection rates broken out by D-size."""
    col_n20 = f"{model_name}_n20"
    col_n50 = f"{model_name}_n50"
    if col_n20 not in obs_df.columns:
        return

    print(f"\n{'─'*60}")
    print(f"DETECTION BY D-SIZE: {model_name}")
    print(f"{'─'*60}")
    print(f"  {'D-size':<8} {'N_obs':>6} {'Det@0.2':>8} {'Det@0.5':>8}")

    for dsize in sorted(obs_df["dsize"].fillna("").astype(str).unique()):
        sub = obs_df[obs_df["dsize"].fillna("").astype(str) == dsize]
        n = len(sub)
        n20 = (sub[col_n20] >= 1).sum()
        n50 = (sub[col_n50] >= 1).sum()
        r20 = f"{100*n20/max(n,1):.0f}%"
        r50 = f"{100*n50/max(n,1):.0f}%"
        print(f"  {str(dsize):<8} {n:>6} {n20:>4}/{n} {r20:>4}  {n50:>4}/{n} {r50:>4}")

    # Overall
    n = len(obs_df)
    n20 = (obs_df[col_n20] >= 1).sum()
    n50 = (obs_df[col_n50] >= 1).sum()
    print(f"  {'ALL':<8} {n:>6} {n20:>4}/{n} {100*n20/max(n,1):.0f}%  {n50:>4}/{n} {100*n50/max(n,1):.0f}%")


# ── Phase 2: Patch extraction ────────────────────────────────────────

def extract_patches(nc, labels_dir, out_dir, date, pair_mode=True, hrrr_path=None):
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
    ]
    if pair_mode:
        cmd += ["--pairs", "--max-pairs", "4"]
    if hrrr_path and Path(hrrr_path).exists():
        cmd += ["--hrrr", str(hrrr_path)]
    subprocess.run(cmd, check=False)


# ── Phase 3: Training ────────────────────────────────────────────────

def train(data_dirs, out_weights, epochs=50, lr=1e-3, resume=None, sar_ch=3):
    """Train on one or more patch directories."""
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
    if resume and resume.exists():
        cmd += ["--resume", str(resume)]
    subprocess.run(cmd, check=False)


# ── Phase 4: Inference ───────────────────────────────────────────────

def run_inference(nc, weights, out_dir, pair_mode=True, sar_ch=3, hrrr_path=None):
    cmd = [
        sys.executable, "scripts/debris_pixel_classifier/v2/full_season_inference.py",
        "--nc", str(nc), "--weights", str(weights),
        "--season", "2025-2026", "--tau", str(TAU),
        "--out-dir", str(out_dir),
        "--no-tiffs", "--stride", "32", "--batch-size", "16",
        "--sar-channels", str(sar_ch),
    ]
    if pair_mode:
        cmd += ["--pairs", "--max-pairs", "4"]
    if hrrr_path and Path(hrrr_path).exists():
        cmd += ["--hrrr", str(hrrr_path)]
    subprocess.run(cmd, check=False)


# ── Phase 5: Observation comparison with F1 ──────────────────────────

def compare_vs_obs(prob_nc_path, label, akdot_paths, obs_gdf, scene_crs, x, y, H, W, transform):
    """Compare a model against observations. Returns per-obs + aggregate metrics."""
    prob_ds = xr.open_dataset(prob_nc_path)

    results = []
    for _, row in obs_gdf.iterrows():
        obs_date = row["avalanche_date"]
        obs_id = row.get("obs_id", row.get("objectid", None))

        # Find path polygon for this obs
        obs_point = row.geometry
        containing = akdot_paths[akdot_paths.contains(obs_point)]
        if len(containing) == 0:
            # Try 150m buffer
            buf = akdot_paths.to_crs("EPSG:3338").copy()
            buf["geometry"] = buf.geometry.buffer(150)
            buf = buf.to_crs("EPSG:4326")
            containing = buf[buf.contains(obs_point)]
        if len(containing) == 0:
            continue

        path_geom = containing.iloc[0].geometry
        path_name = containing.iloc[0]["name"]

        try:
            path_mask = ~rasterio.features.geometry_mask(
                [path_geom], out_shape=(H, W), transform=transform, all_touched=True,
            )
        except Exception:
            continue
        if path_mask.sum() == 0:
            continue

        # Find closest CNN date
        cnn_times = pd.DatetimeIndex(prob_ds.time.values)
        time_diffs = np.abs(cnn_times - obs_date)
        ci = time_diffs.argmin()
        if time_diffs[ci].days > 6:
            continue

        prob_map = prob_ds["debris_probability"].isel(time=ci).values
        path_probs = prob_map[path_mask]

        results.append({
            "obs_id": obs_id,
            "date": obs_date.strftime("%Y-%m-%d"),
            "path_name": path_name,
            "dsize": row.get("dsize", ""),
            "n_path_px": int(path_mask.sum()),
            f"{label}_mean": float(np.mean(path_probs)),
            f"{label}_max": float(np.max(path_probs)),
            f"{label}_n20": int((path_probs > 0.2).sum()),
            f"{label}_n50": int((path_probs > 0.5).sum()),
        })

    prob_ds.close()
    return pd.DataFrame(results)


def compute_path_f1(prob_nc_path, akdot_paths_in_scene, obs_gdf, x, y, H, W, transform, thresholds=[0.2, 0.5]):
    """Compute path-level F1: for each SAR date, which paths have detections vs observations."""
    prob_ds = xr.open_dataset(prob_nc_path)
    cnn_times = pd.DatetimeIndex(prob_ds.time.values)

    all_results = []
    for ci, cnn_date in enumerate(cnn_times):
        prob_map = prob_ds["debris_probability"].isel(time=ci).values

        for _, path_row in akdot_paths_in_scene.iterrows():
            path_name = path_row["name"]
            try:
                path_mask = ~rasterio.features.geometry_mask(
                    [path_row.geometry], out_shape=(H, W), transform=transform, all_touched=True,
                )
            except Exception:
                continue
            if path_mask.sum() == 0:
                continue

            path_probs = prob_map[path_mask]

            # Was there an observation on this path within +-6 days?
            obs_nearby = obs_gdf[
                (np.abs((obs_gdf["avalanche_date"] - cnn_date).dt.days) <= 6)
            ]
            # Check if any obs point falls in this path (with buffer)
            has_obs = False
            for _, obs_row in obs_nearby.iterrows():
                if path_row.geometry.contains(obs_row.geometry):
                    has_obs = True
                    break

            for thresh in thresholds:
                has_detection = int((path_probs > thresh).sum()) > 0
                all_results.append({
                    "cnn_date": str(cnn_date)[:10],
                    "path_name": path_name,
                    "threshold": thresh,
                    "has_obs": has_obs,
                    "has_detection": has_detection,
                    "n_above": int((path_probs > thresh).sum()),
                })

    prob_ds.close()
    return pd.DataFrame(all_results)


def print_f1_report(f1_df, label):
    """Print F1 and FP rate from path-level results."""
    print(f"\n{'='*80}")
    print(f"PATH-LEVEL F1 REPORT: {label}")
    print(f"{'='*80}")
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
        print(f"    False positive rate={fpr:.4f} ({fp} false path-date detections)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load season dataset ──────────────────────────────────────────
    log.info("Loading 2025-2026 season dataset...")
    ds = load_netcdf_to_dataset(NC)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    times = pd.DatetimeIndex(ds.time.values)
    in_season = times[(times.month >= 11) | (times.month <= 4)]
    all_dates = sorted(set(t.strftime("%Y-%m-%d") for t in in_season))
    log.info("  %d in-season SAR dates", len(all_dates))
    scene_box = box(*ds.rio.bounds())

    # ── Load all observations (AKDOT + AKRR) ────────────────────────
    log.info("Loading observations...")

    # AKDOT 2025-2026
    obs_all = pd.read_csv(str(AKDOT_OBS))
    obs_all["avalanche_date"] = pd.to_datetime(obs_all["avalanche_date"])
    obs_2526 = obs_all[(obs_all["avalanche_date"] >= "2025-11-01") & (obs_all["avalanche_date"] <= "2026-04-30")]
    obs_2526 = obs_2526[obs_2526["latitude"].notna()].copy()
    akdot_2526_gdf = gpd.GeoDataFrame(
        obs_2526, geometry=gpd.points_from_xy(obs_2526.longitude, obs_2526.latitude), crs="EPSG:4326",
    )
    akdot_2526_gdf = akdot_2526_gdf[akdot_2526_gdf.within(scene_box)]
    log.info("  AKDOT 2025-2026: %d obs in scene", len(akdot_2526_gdf))

    # AKRR
    akrr_gdf = parse_akrr_obs()

    # Merge AKDOT + AKRR for 2025-2026
    all_obs_gdf = merge_obs_sources(akdot_2526_gdf, akrr_gdf)

    # Temporal train/val split
    train_obs, val_obs, split_date = temporal_train_val_split(all_obs_gdf)
    log.info("Train obs: %d, Val obs: %d, Split date: %s",
             len(train_obs), len(val_obs), str(split_date)[:10] if split_date else "N/A")

    # ── Also load 2024-2025 observations for extra training data ─────
    train_obs_2425 = gpd.GeoDataFrame()
    if NC_2425.exists():
        obs_2425 = obs_all[(obs_all["avalanche_date"] >= "2024-11-01") & (obs_all["avalanche_date"] <= "2025-04-30")]
        obs_2425 = obs_2425[obs_2425["latitude"].notna()].copy()
        if len(obs_2425) > 0:
            akdot_2425_gdf = gpd.GeoDataFrame(
                obs_2425, geometry=gpd.points_from_xy(obs_2425.longitude, obs_2425.latitude), crs="EPSG:4326",
            )
            akdot_2425_gdf = akdot_2425_gdf[akdot_2425_gdf.within(scene_box)]
            if len(akdot_2425_gdf) > 0:
                # Use all 2024-2025 obs for training (not validation — different season)
                train_obs_2425 = akdot_2425_gdf.copy()
                log.info("  AKDOT 2024-2025: %d obs in scene (all for training)", len(train_obs_2425))

    # Score each date by nearby obs for auto-label date selection
    date_scores = []
    for d in all_dates:
        t = pd.Timestamp(d)
        nearby = akdot_2526_gdf[(akdot_2526_gdf["avalanche_date"] >= t - pd.Timedelta(days=6)) &
                                 (akdot_2526_gdf["avalanche_date"] <= t + pd.Timedelta(days=6))]
        date_scores.append({"date": d, "n_obs": len(nearby), "month": t.month})
    score_df = pd.DataFrame(date_scores)

    # Select: top active dates (spaced 12+ days) + quiet dates
    selected = set()
    used = set()
    def ok(d, gap=12):
        return all(abs((pd.Timestamp(d) - pd.Timestamp(s)).days) >= gap for s in used)

    for _, r in score_df.sort_values("n_obs", ascending=False).iterrows():
        if r["n_obs"] >= 5 and ok(r["date"]):
            selected.add(r["date"])
            used.add(r["date"])
        if sum(1 for d in selected if score_df[score_df["date"] == d]["n_obs"].iloc[0] >= 5) >= 6:
            break

    for _, r in score_df[score_df["n_obs"] <= 1].iterrows():
        if ok(r["date"]):
            selected.add(r["date"])
            used.add(r["date"])
        if sum(1 for d in selected if score_df[score_df["date"] == d]["n_obs"].iloc[0] <= 1) >= 3:
            break

    dates = sorted(selected)
    log.info("Selected %d dates for auto-labeling: %s", len(dates), dates)
    for d in dates:
        row = score_df[score_df["date"] == d].iloc[0]
        log.info("  %s  obs=%d", d, row["n_obs"])

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Smart auto-labeling
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 1: Smart auto-labeling (elev-banded Otsu + CR suppression) ===")

    # Build glacier mask from RGI
    glacier_mask = build_glacier_mask(ds)

    auto_dir = OUT / "auto_labels"
    total_polys = 0
    date_poly_counts = {}
    for date_str in dates:
        log.info("  Date: %s", date_str)
        ds, n = smart_auto_label(ds, date_str, auto_dir, tau=TAU,
                                  glacier_mask=glacier_mask)
        total_polys += n
        date_poly_counts[date_str] = n
    log.info("Total auto polygons: %d across %d dates", total_polys, len(dates))

    # Quiet-period sanity check
    log.info("=== Quiet-period check ===")
    for d_str in dates:
        row = score_df[score_df["date"] == d_str].iloc[0]
        n_polys = date_poly_counts.get(d_str, 0)
        flag = " *** HIGH" if row["n_obs"] <= 1 and n_polys > 10 else ""
        log.info("  %s  obs=%d  polys=%d%s", d_str, row["n_obs"], n_polys, flag)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1b: Observation-guided labeling (TRAIN SPLIT ONLY)
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 1b: Observation-guided labeling (train split only) ===")
    akdot_paths = gpd.read_file(str(AKDOT_PATHS))
    akrr_paths = gpd.read_file(str(AKRR_PATHS)) if AKRR_PATHS.exists() else gpd.GeoDataFrame()
    # Combine path polygons for matching
    all_paths = akdot_paths.copy()
    if len(akrr_paths) > 0:
        existing_names = set(n.lower() for n in akdot_paths["name"].values)
        new_akrr = akrr_paths[~akrr_paths["name"].str.lower().isin(existing_names)]
        if len(new_akrr) > 0:
            all_paths = pd.concat([all_paths, new_akrr[["name", "geometry"]]], ignore_index=True)
            all_paths = gpd.GeoDataFrame(all_paths, crs=akdot_paths.crs)
            log.info("  Added %d unique AKRR paths to path inventory", len(new_akrr))

    obs_guided_dir = OUT / "obs_guided_labels"

    # 2025-2026 train split
    ds, obs_poly_counts_2526 = obs_guided_label(
        ds, train_obs, all_paths, obs_guided_dir, tau=TAU,
        d_threshold=1.5, min_component_px=3,
    )
    total_obs_polys = sum(obs_poly_counts_2526.values())
    log.info("2025-2026 obs-guided polygons: %d across %d dates",
             total_obs_polys, len(obs_poly_counts_2526))

    # 2024-2025 obs-guided labels (if available)
    obs_guided_dir_2425 = OUT / "obs_guided_labels_2425"
    obs_poly_counts_2425 = {}
    if len(train_obs_2425) > 0 and NC_2425.exists():
        log.info("Loading 2024-2025 season for obs-guided labeling...")
        ds_2425 = load_netcdf_to_dataset(NC_2425)
        if not np.issubdtype(ds_2425["time"].dtype, np.datetime64):
            ds_2425["time"] = pd.DatetimeIndex(ds_2425["time"].values)
        if any(var.chunks is not None for var in ds_2425.variables.values()):
            ds_2425 = ds_2425.load()
        ds_2425, obs_poly_counts_2425 = obs_guided_label(
            ds_2425, train_obs_2425, all_paths, obs_guided_dir_2425, tau=TAU,
            d_threshold=1.5, min_component_px=3,
        )
        total_2425 = sum(obs_poly_counts_2425.values())
        log.info("2024-2025 obs-guided polygons: %d across %d dates",
                 total_2425, len(obs_poly_counts_2425))
        del ds_2425  # free memory

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Extract patches
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 2: Extract patches ===")
    human_patch_dir = OUT / "patches" / "human"
    auto_patch_dir = OUT / "patches" / "auto"
    obs_patch_dir = OUT / "patches" / "obs_guided"
    obs_patch_dir_2425 = OUT / "patches" / "obs_guided_2425"

    # Human: 2026-02-14
    log.info("Extracting human patches...")
    extract_patches(NC, HUMAN_LABELS, human_patch_dir, "2026-02-14", pair_mode=True)

    # Auto: all dates with labels
    log.info("Extracting auto patches...")
    for gpkg_file in sorted(auto_dir.glob("avalanche_labels_*.gpkg")):
        date = gpkg_file.stem.replace("avalanche_labels_", "")
        extract_patches(NC, auto_dir, auto_patch_dir, date, pair_mode=True)

    # Obs-guided 2025-2026
    log.info("Extracting obs-guided patches (2025-2026)...")
    for gpkg_file in sorted(obs_guided_dir.glob("avalanche_labels_*.gpkg")):
        date = gpkg_file.stem.replace("avalanche_labels_", "")
        extract_patches(NC, obs_guided_dir, obs_patch_dir, date, pair_mode=True)

    # Obs-guided 2024-2025
    if obs_poly_counts_2425:
        log.info("Extracting obs-guided patches (2024-2025)...")
        for gpkg_file in sorted(obs_guided_dir_2425.glob("avalanche_labels_*.gpkg")):
            date = gpkg_file.stem.replace("avalanche_labels_", "")
            extract_patches(NC_2425, obs_guided_dir_2425, obs_patch_dir_2425, date, pair_mode=True)

    # Assign confidence
    log.info("Assigning confidence...")
    for pdir in [human_patch_dir, auto_patch_dir, obs_patch_dir, obs_patch_dir_2425]:
        if pdir.exists():
            subprocess.run([sys.executable, "scripts/debris_pixel_classifier/v2/assign_confidence.py",
                            "--patches-dir", str(pdir)], check=False)

    # Boost confidence for obs-guided patches (confirmed events)
    for obs_dir in [obs_patch_dir, obs_patch_dir_2425]:
        if not obs_dir.exists():
            continue
        for lf in obs_dir.rglob("labels.json"):
            with open(lf) as f:
                labels = json.load(f)
            changed = False
            for info in labels.values():
                if info.get("label") == 1:
                    old = info.get("confidence", 0.5)
                    info["confidence"] = round(max(old, 0.6), 4)
                    changed = True
            if changed:
                with open(lf, "w") as f:
                    json.dump(labels, f, indent=2)

    # Count patches
    def count_patches(d):
        p = n = 0
        if not d.exists():
            return 0, 0
        for lf in d.rglob("labels.json"):
            with open(lf) as f:
                lab = json.load(f)
            for v in lab.values():
                if v.get("label") == 1: p += 1
                elif v.get("label") == 0: n += 1
        return p, n

    hp, hn = count_patches(human_patch_dir)
    ap, an = count_patches(auto_patch_dir)
    op, on = count_patches(obs_patch_dir)
    op2, on2 = count_patches(obs_patch_dir_2425)
    sp, sn = count_patches(SNFAC_PATCHES)
    log.info("Human patches:           %d pos, %d neg", hp, hn)
    log.info("Auto patches:            %d pos, %d neg", ap, an)
    log.info("Obs-guided 2025-2026:    %d pos, %d neg", op, on)
    log.info("Obs-guided 2024-2025:    %d pos, %d neg", op2, on2)
    log.info("SNFAC pretrain patches:  %d pos, %d neg", sp, sn)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: 3-Stage Training
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 3: 3-Stage Training ===")
    weights_dir = OUT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Collect obs-guided patch dirs
    obs_dirs = [obs_patch_dir]
    if obs_patch_dir_2425.exists() and count_patches(obs_patch_dir_2425)[0] > 0:
        obs_dirs.append(obs_patch_dir_2425)

    # ── Stage 1: Pretrain on SNFAC (Idaho, general debris features) ──
    stage1_weights = weights_dir / "stage1_snfac.pt"
    log.info("Stage 1: Pretrain on SNFAC (%s)...", SNFAC_PATCHES)
    train(SNFAC_PATCHES, stage1_weights, epochs=50, lr=1e-3, sar_ch=3)

    # ── Stage 2: Bridge on CNFAIC auto + obs-guided train split ──────
    stage2_weights = weights_dir / "stage2_bridge.pt"
    stage2_dirs = [auto_patch_dir] + obs_dirs
    log.info("Stage 2: Bridge on CNFAIC auto + obs-guided (%d dirs)...", len(stage2_dirs))
    train(stage2_dirs, stage2_weights, epochs=30, lr=5e-4, resume=stage1_weights, sar_ch=3)

    # ── Stage 3: Finetune on CNFAIC human + obs-guided train split ───
    stage3_weights = weights_dir / "stage3_finetune.pt"
    stage3_dirs = [human_patch_dir] + obs_dirs
    log.info("Stage 3: Finetune on CNFAIC human + obs-guided (%d dirs)...", len(stage3_dirs))
    train(stage3_dirs, stage3_weights, epochs=20, lr=1e-4, resume=stage2_weights, sar_ch=3)

    # ── Baselines ────────────────────────────────────────────────────
    # Baseline A: Human-only (CNFAIC only, no pretrain)
    human_only_weights = weights_dir / "human_only.pt"
    log.info("Baseline A: Human-only...")
    train(human_patch_dir, human_only_weights, epochs=50, lr=1e-3, sar_ch=3)

    # Baseline B: 2-stage (CNFAIC auto → human, no SNFAC)
    b2_stage1 = weights_dir / "b2_auto_pretrain.pt"
    b2_stage2 = weights_dir / "b2_auto_finetune.pt"
    log.info("Baseline B: 2-stage (CNFAIC auto → human)...")
    train(auto_patch_dir, b2_stage1, epochs=50, lr=1e-3, sar_ch=3)
    train(human_patch_dir, b2_stage2, epochs=20, lr=1e-4, resume=b2_stage1, sar_ch=3)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Season inference
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 4: Season inference ===")

    model_variants = [
        ("3stage", stage3_weights),
        ("2stage", b2_stage2),
        ("human_only", human_only_weights),
    ]

    for name, weights in model_variants:
        if not weights.exists():
            log.warning("No weights for %s, skipping", name)
            continue
        inf_dir = OUT / f"inference_{name}"
        log.info("Running inference: %s", name)
        run_inference(NC, weights, inf_dir, pair_mode=True, sar_ch=3)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Validation (held-out observations only)
    # ══════════════════════════════════════════════════════════════════
    log.info("=== PHASE 5: Validation (held-out obs after %s) ===",
             str(split_date)[:10] if split_date else "N/A")

    # Use val_obs for comparison (the held-out 40%)
    akdot_in_scene = akdot_paths[akdot_paths.intersects(scene_box)].copy()
    if len(akrr_paths) > 0:
        akrr_in_scene = akrr_paths[akrr_paths.intersects(scene_box)].copy()
        existing_names = set(n.lower() for n in akdot_in_scene["name"].values)
        new_akrr = akrr_in_scene[~akrr_in_scene["name"].str.lower().isin(existing_names)]
        if len(new_akrr) > 0:
            akdot_in_scene = pd.concat([akdot_in_scene, new_akrr[["name", "geometry"]]], ignore_index=True)
            akdot_in_scene = gpd.GeoDataFrame(akdot_in_scene, crs=akdot_paths.crs)

    log.info("Val obs: %d, Paths in scene: %d", len(val_obs), len(akdot_in_scene))

    # Grid info
    x, y = ds.x.values, ds.y.values
    H, W = len(y), len(x)
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    transform = from_bounds(
        float(x.min()) - dx/2, float(y.min()) - dy/2,
        float(x.max()) + dx/2, float(y.max()) + dy/2, W, H,
    )

    # Compare each model variant against held-out observations
    all_obs_results = {}
    model_names = [name for name, _ in model_variants]

    for name in model_names:
        prob_nc = OUT / f"inference_{name}" / "season_v2_debris_probabilities.nc"
        if not prob_nc.exists():
            log.warning("No inference for %s", name)
            continue

        log.info("Comparing %s vs held-out observations...", name)
        obs_df = compare_vs_obs(prob_nc, name, akdot_in_scene, val_obs,
                                "EPSG:4326", x, y, H, W, transform)
        all_obs_results[name] = obs_df

        if len(obs_df) > 0:
            n20 = (obs_df[f"{name}_n20"] >= 1).sum()
            n50 = (obs_df[f"{name}_n50"] >= 1).sum()
            n_total = len(obs_df)
            print(f"\n{name}: {n20}/{n_total} detected (>0.2), {n50}/{n_total} detected (>0.5)")
            print_dsize_report(obs_df, name)

        # Path-level F1
        f1_df = compute_path_f1(prob_nc, akdot_in_scene, val_obs, x, y, H, W, transform)
        print_f1_report(f1_df, name)
        f1_df.to_csv(OUT / f"path_f1_{name}.csv", index=False)

    # Merge all obs results into one comparison table
    if len(all_obs_results) >= 2:
        base_name = model_names[0]
        merged = all_obs_results[base_name].copy() if base_name in all_obs_results else pd.DataFrame()
        for name in model_names[1:]:
            if name in all_obs_results and len(merged) > 0:
                other = all_obs_results[name].drop(
                    columns=["date", "path_name", "dsize", "n_path_px"], errors="ignore",
                )
                merged = merged.merge(other, on="obs_id", how="outer")
        if len(merged) > 0:
            merged.to_csv(OUT / "obs_comparison.csv", index=False)

            print(f"\n{'='*80}")
            print("OVERALL COMPARISON (held-out validation obs)")
            print(f"{'='*80}")
            for name in model_names:
                col20 = f"{name}_n20"
                col50 = f"{name}_n50"
                if col20 in merged.columns:
                    n = len(merged[merged[col20].notna()])
                    n20 = (merged[col20] >= 1).sum()
                    n50 = (merged[col50] >= 1).sum()
                    total_px = merged[col20].sum()
                    print(f"  {name:20s}  detected(>0.2)={n20}/{n}  detected(>0.5)={n50}/{n}  total_px={total_px:.0f}")

    log.info("Results saved to %s", OUT)


if __name__ == "__main__":
    main()
