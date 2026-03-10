#!/usr/bin/env python3
"""Plot side-by-side d_empirical and CNN debris probability for missed D3+ avalanches.

Left panel:  d_empirical (backscatter change) at the observation date
Right panel: CNN debris_probability at the nearest time step

Both panels show FlowPy track boundary (green) and observation location (cyan star).
A short diagnosis of why the detection was missed is annotated at the bottom.

Saves to local/issw/figures/missed_d3s/
"""

import logging
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from sarvalanche.io.dataset import load_netcdf_to_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
NC_DIR = ROOT / "local/issw/netcdfs"
OBS_DIR = ROOT / "local/issw/observations"
OUT_DIR = ROOT / "local/issw/figures/missed_d3s"
CSV_PATH = ROOT / "local/issw/snfac_obs_2021_2025.csv"

PLOT_BUFFER_DEG = 0.04  # ~4 km


def find_cnn_nc(cnn_source: str) -> Path | None:
    """Map CNN source name to the debris_probability NetCDF."""
    m = re.match(r"^(.+)_(\d{4}-\d{4})$", cnn_source)
    if m:
        zone_name = m.group(1)
        season = m.group(2)
        nc = NC_DIR / zone_name / f"v2_season_inference_{season}" / "season_v2_debris_probabilities.nc"
        if nc.exists():
            return nc
    zone_name = cnn_source if not m else m.group(1)
    nc = NC_DIR / zone_name / "v2_season_inference" / "season_v2_debris_probabilities.nc"
    return nc if nc.exists() else None


def find_season_nc(cnn_source: str) -> Path | None:
    """Map CNN source name to the season dataset NetCDF."""
    m = re.match(r"^(.+)_(\d{4}-\d{4})$", cnn_source)
    if m:
        zone_name = m.group(1)
        season = m.group(2)
    else:
        zone_name = cnn_source
        season = None
    zone_dir = NC_DIR / zone_name
    if not zone_dir.is_dir():
        return None
    if season:
        for nc in zone_dir.glob(f"season_{season}*.nc"):
            return nc
        for nc in zone_dir.glob(f"season_{season.replace('-', '_')}*.nc"):
            return nc
    ncs = sorted(zone_dir.glob("season_*.nc"))
    return ncs[0] if ncs else None


def compute_d_empirical_for_date(ds, obs_date, tau_days=6):
    """Compute d_empirical at the given observation date."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(obs_date)
    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    try:
        p_emp, d_emp = calculate_empirical_backscatter_probability(
            ds, ref_ts,
            use_agreement_boosting=True,
            agreement_strength=0.8,
            min_prob_threshold=0.2,
            tau_days=tau_days,
        )
    except (ValueError, KeyError) as e:
        log.warning("d_empirical failed: %s", e)
        return None
    return d_emp


def make_path_mask(path_geom, y_coords, x_coords):
    """Rasterize a path geometry onto the given grid. Returns boolean (ny, nx)."""
    from rasterio.transform import from_bounds
    import rasterio.features
    ny, nx = len(y_coords), len(x_coords)
    left, right = float(x_coords.min()), float(x_coords.max())
    bottom, top = float(y_coords.min()), float(y_coords.max())
    transform = from_bounds(left, bottom, right, top, nx, ny)
    mask = rasterio.features.geometry_mask(
        [path_geom], out_shape=(ny, nx), transform=transform, invert=True,
    )
    return mask


def get_feature_stats(ds, path_geom, obs_date):
    """Extract CNN-relevant feature statistics inside the FlowPy path.

    Returns a dict of feature diagnostics for the path area.
    """
    stats = {}
    y_coords = ds.y.values
    x_coords = ds.x.values

    try:
        mask = make_path_mask(path_geom, y_coords, x_coords)
    except Exception:
        return stats

    n_path_px = int(mask.sum())
    if n_path_px == 0:
        return stats
    stats["n_path_px"] = n_path_px

    # Static terrain features
    for var in ["slope", "fcf", "cell_counts", "release_zones", "dem"]:
        if var in ds.data_vars:
            vals = ds[var].values
            if vals.ndim == 2:
                inside = vals[mask]
                inside = inside[np.isfinite(inside)]
                if len(inside) > 0:
                    stats[f"{var}_mean"] = float(np.mean(inside))
                    if var == "slope":
                        # Convert from radians to degrees if needed
                        mean_val = float(np.mean(inside))
                        stats["slope_mean_deg"] = mean_val if mean_val > 1.5 else np.degrees(mean_val)

    # Runout angle
    if "runout_angle" in ds.data_vars:
        ra = ds["runout_angle"].values
        if ra.ndim == 2:
            inside = ra[mask]
            inside = inside[np.isfinite(inside) & (inside > 0)]
            if len(inside) > 0:
                stats["runout_angle_mean"] = float(np.mean(inside))

    # Water mask
    if "water_mask" in ds.data_vars:
        wm = ds["water_mask"].values
        if wm.ndim == 2:
            stats["water_frac"] = float(wm[mask].mean())

    # d_empirical (if already computed)
    if "d_empirical" in ds.data_vars:
        de = ds["d_empirical"].values
        if de.ndim == 2:
            inside = de[mask]
            inside = inside[np.isfinite(inside)]
            if len(inside) > 0:
                stats["d_emp_mean"] = float(np.mean(inside))
                stats["d_emp_max"] = float(np.max(inside))
                stats["d_emp_min"] = float(np.min(inside))

    # Per-track SAR change maps
    import re as _re
    pattern = _re.compile(r"^d_(\d+)_(V[VH])_empirical$")
    for var in ds.data_vars:
        m = pattern.match(var)
        if m:
            arr = ds[var].values
            if arr.ndim == 2:
                inside = arr[mask]
                inside = inside[np.isfinite(inside)]
                if len(inside) > 0:
                    stats[f"{var}_mean"] = float(np.mean(inside))

    # w_resolution inside path
    if "w_resolution" in ds.data_vars:
        wr = ds["w_resolution"].values
        # May be (track, y, x) or (y, x)
        if wr.ndim == 2:
            inside = wr[mask]
            inside = inside[np.isfinite(inside)]
            if len(inside) > 0:
                stats["w_res_mean"] = float(np.mean(inside))

    return stats


def diagnose_miss(row, feat_stats=None) -> str:
    """Generate a diagnosis of why this observation was missed.

    Combines detection-level stats with actual CNN input feature values
    inside the FlowPy path to pinpoint the likely cause.
    """
    spatial = row["spatial_detected"]
    max_prob = row["max_prob_path"]
    max_bg = row["max_prob_bg"]
    onset_diff = row["onset_date_diff_days_min"]
    n_cand = row.get("n_candidates_in_path", 0)
    feat = feat_stats or {}

    primary = []     # main reason for miss
    context = []     # contributing feature-level factors

    # ── Primary detection-level reason ────────────────────────────────
    if spatial and not np.isnan(onset_diff) and onset_diff > 6:
        primary.append(f"onset {onset_diff:.0f}d off (spatial OK, timing wrong)")
    elif not spatial:
        if max_prob < 0.05:
            primary.append(f"CNN prob near zero in path ({max_prob:.3f})")
        elif max_prob < 0.3:
            primary.append(f"weak CNN signal in path ({max_prob:.2f})")
        elif not np.isnan(max_bg) and max_bg >= max_prob:
            primary.append(f"bg prob ({max_bg:.2f}) >= path ({max_prob:.2f})")
        else:
            primary.append("spatial criterion failed")

    if n_cand == 0 and max_prob > 0.3:
        primary.append("no onset candidates despite prob")

    # ── Feature-level context (always include if available) ───────────
    slope_deg = feat.get("slope_mean_deg")
    if slope_deg is not None:
        if slope_deg < 20:
            context.append(f"slope={slope_deg:.0f}° (low, runout zone)")
        elif slope_deg > 55:
            context.append(f"slope={slope_deg:.0f}° (cliff/rock)")
        else:
            context.append(f"slope={slope_deg:.0f}°")

    fcf_mean = feat.get("fcf_mean")
    if fcf_mean is not None:
        if fcf_mean > 40:
            context.append(f"fcf={fcf_mean:.0f}% (forested)")
        elif fcf_mean > 10:
            context.append(f"fcf={fcf_mean:.0f}%")

    cc_mean = feat.get("cell_counts_mean")
    if cc_mean is not None:
        if cc_mean < 1:
            context.append("cell_counts<1 (no FlowPy convergence)")
        else:
            context.append(f"cell_counts={cc_mean:.0f}")

    d_emp_max = feat.get("d_emp_max")
    d_emp_mean = feat.get("d_emp_mean")
    if d_emp_max is not None:
        if abs(d_emp_max) < 0.5:
            context.append(f"d_emp flat (max={d_emp_max:.2f}, no SAR signal)")
        elif d_emp_mean is not None and d_emp_mean < -1.0:
            context.append(f"d_emp={d_emp_mean:.1f} (neg change, wet snow?)")
        elif d_emp_max > 1.0:
            context.append(f"d_emp max={d_emp_max:.1f} (has SAR signal)")

    w_res = feat.get("w_res_mean")
    if w_res is not None and w_res < 0.3:
        context.append(f"w_res={w_res:.2f} (poor SAR geometry)")

    water_frac = feat.get("water_frac")
    if water_frac is not None and water_frac > 0.1:
        context.append(f"water={water_frac:.0%}")

    # ── Observed aspect vs DEM aspect mismatch ──────────────────────
    obs_aspect = row.get("aspect", "")
    if pd.notna(obs_aspect) and obs_aspect:
        context.append(f"reported aspect={obs_aspect}")

    # ── Combine ───────────────────────────────────────────────────────
    if not primary:
        primary.append("marginal case")

    parts = primary
    if context:
        parts = primary + [" | " + ", ".join(context)]
    return "".join(parts)


def compute_hillshade(dem_2d, azimuth=315, altitude=45):
    """Simple hillshade from a 2D DEM array."""
    dy, dx = np.gradient(dem_2d)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    shade = (
        np.sin(alt_rad) * np.cos(np.arctan(np.sqrt(dx**2 + dy**2)))
        + np.cos(alt_rad) * np.sin(np.arctan(np.sqrt(dx**2 + dy**2)))
        * np.cos(az_rad - np.arctan2(-dx, dy))
    )
    shade = np.clip(shade, 0, 1)
    return shade


def plot_observation(
    d_emp_clip, d_emp_extent,
    prob_clip, prob_extent,
    dem_clip,
    obs_lng, obs_lat,
    path_geom,
    obs_id, obs_date, cnn_date,
    d_size, avy_type, zone_name,
    obs_aspect, obs_elev,
    max_prob, onset_diff, diagnosis,
    out_path,
):
    """Create side-by-side plot with hillshade underlay on CNN panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Left: d_empirical
    if d_emp_clip is not None:
        im1 = ax1.imshow(
            d_emp_clip, origin="upper", extent=d_emp_extent,
            cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
        )
        fig.colorbar(im1, ax=ax1, shrink=0.7, label="d_empirical (dB)")
        ax1.set_title(f"d_empirical — {cnn_date}", fontsize=11)
    else:
        ax1.text(0.5, 0.5, "d_empirical\nnot available", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title("d_empirical — N/A", fontsize=11)

    # Right: CNN probability with hillshade underlay
    # Draw hillshade first
    if dem_clip is not None:
        hs = compute_hillshade(dem_clip)
        ax2.imshow(hs, origin="upper", extent=prob_extent,
                   cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    # Make CNN probs < 0.1 transparent
    prob_rgba = plt.cm.hot_r(prob_clip)  # (ny, nx, 4) RGBA
    prob_rgba[..., 3] = np.where(prob_clip < 0.1, 0.0, 0.85)  # alpha=0 below 0.1

    im2 = ax2.imshow(
        prob_rgba, origin="upper", extent=prob_extent, interpolation="nearest",
    )
    # Add a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax2, shrink=0.7, label="CNN debris probability")
    ax2.set_title(f"CNN probability — {cnn_date}", fontsize=11)

    # Overlay track and point on both axes
    for ax in (ax1, ax2):
        if path_geom is not None and not path_geom.is_empty:
            track_gdf = gpd.GeoDataFrame(geometry=[path_geom], crs="EPSG:4326")
            track_gdf.boundary.plot(ax=ax, color="lime", linewidth=1.5, label="FlowPy track")
        ax.scatter(obs_lng, obs_lat, c="cyan", s=150, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=10, label="Observation")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right", fontsize=8)

    avy_str = avy_type if pd.notna(avy_type) else "unknown"
    aspect_str = f", aspect={obs_aspect}" if pd.notna(obs_aspect) else ""
    elev_str = f", elev={obs_elev:.0f}ft" if pd.notna(obs_elev) else ""
    fig.suptitle(
        f"Missed D{d_size} ({avy_str}{aspect_str}{elev_str}) — {zone_name}\n"
        f"max_prob={max_prob:.3f}, onset_diff={onset_diff:.0f}d",
        fontsize=13, fontweight="bold",
    )
    # Diagnosis annotation at bottom
    fig.text(0.5, 0.01, f"Why missed: {diagnosis}", ha="center", fontsize=10,
             style="italic", color="darkred",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clip_to_extent(data_2d, y_coords, x_coords, obs_lng, obs_lat, buf):
    """Clip 2D array to buffer around point. Returns (clipped_array, extent) or (None, None)."""
    x_mask = (x_coords >= obs_lng - buf) & (x_coords <= obs_lng + buf)
    y_mask = (y_coords >= obs_lat - buf) & (y_coords <= obs_lat + buf)
    if x_mask.sum() == 0 or y_mask.sum() == 0:
        return None, None
    xi = np.where(x_mask)[0]
    yi = np.where(y_mask)[0]
    clip = data_2d[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
    x_sub = x_coords[xi[0]:xi[-1]+1]
    y_sub = y_coords[yi[0]:yi[-1]+1]
    extent = [float(x_sub.min()), float(x_sub.max()),
              float(y_sub.min()), float(y_sub.max())]
    return clip, extent


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(OBS_DIR / "comparison_summary.csv")
    obs_df = pd.read_csv(CSV_PATH)
    merged = comp.merge(
        obs_df[["id", "avalanche_type", "d_size", "observer_type", "location_name", "aspect", "elevation"]],
        left_on="obs_id", right_on="id", how="left", suffixes=("", "_obs"),
    )

    missed = merged[
        (merged["d_size"] >= 3.0)
        & (~merged["detected"])
        & merged["onset_date_diff_days_min"].notna()
    ].sort_values("d_size", ascending=False).copy()

    log.info("Found %d missed D3+ observations", len(missed))

    # Load FlowPy paths
    all_paths = gpd.read_file(OBS_DIR / "all_flowpy_paths.gpkg")
    all_paths["geometry"] = all_paths.geometry.buffer(500)
    if all_paths.crs and str(all_paths.crs) != "EPSG:4326":
        all_paths = all_paths.to_crs("EPSG:4326")

    # Caches
    cnn_cache = {}      # cnn_nc_path -> (prob_data, times, y, x)
    ds_cache = {}       # season_nc_path -> dataset
    demp_cache = {}     # (season_nc_path, date_str) -> (d_emp_values, y, x)

    n_plotted = 0
    for _, row in missed.iterrows():
        obs_id = row["obs_id"]
        obs_date = row["date"]
        cnn_source = row["cnn_source"]
        d_size = row["d_size"]
        avy_type = row.get("avalanche_type", "")
        zone_name = row.get("zone_name", "")
        max_prob = row["max_prob_path"]
        onset_diff = row["onset_date_diff_days_min"]
        obs_lng = row["lng"]
        obs_lat = row["lat"]

        safe_id = str(obs_id).replace("/", "_")
        out_path = OUT_DIR / f"D{d_size}_{obs_date}_{safe_id}.png"
        if out_path.exists():
            n_plotted += 1
            continue

        # ── CNN probability ───────────────────────────────────────────
        cnn_nc = find_cnn_nc(cnn_source)
        if cnn_nc is None:
            log.warning("No CNN NC for %s", obs_id)
            continue

        nc_key = str(cnn_nc)
        if nc_key not in cnn_cache:
            log.info("Loading CNN: %s", cnn_nc.name)
            ds_cnn = xr.open_dataset(cnn_nc)
            prob_da = ds_cnn["debris_probability"]
            cnn_cache[nc_key] = (
                prob_da.values,
                pd.DatetimeIndex(prob_da.time.values),
                prob_da.y.values,
                prob_da.x.values,
            )
            ds_cnn.close()

        prob_data, cnn_times, cnn_y, cnn_x = cnn_cache[nc_key]
        obs_ts = pd.Timestamp(obs_date)
        tidx = int(np.argmin(np.abs(cnn_times - obs_ts)))
        cnn_date = str(cnn_times[tidx].date())
        prob_2d = prob_data[tidx]

        prob_clip, prob_extent = clip_to_extent(prob_2d, cnn_y, cnn_x, obs_lng, obs_lat, PLOT_BUFFER_DEG)
        if prob_clip is None:
            continue

        # ── d_empirical (at CNN date so both panels match) ─────────────
        season_nc = find_season_nc(cnn_source)
        d_emp_clip, d_emp_extent = None, None

        if season_nc is not None:
            snc_key = str(season_nc)
            demp_key = (snc_key, cnn_date)  # use CNN acquisition date, not obs date

            if demp_key not in demp_cache:
                # Load dataset if needed
                if snc_key not in ds_cache:
                    log.info("Loading season: %s", season_nc.name)
                    ds = load_netcdf_to_dataset(season_nc)
                    if not np.issubdtype(ds["time"].dtype, np.datetime64):
                        ds["time"] = pd.DatetimeIndex(ds["time"].values)
                    if "w_resolution" not in ds.data_vars:
                        from sarvalanche.weights.local_resolution import get_local_resolution_weights
                        ds["w_resolution"] = get_local_resolution_weights(ds["anf"])
                        ds["w_resolution"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}
                    if "p_fcf" not in ds.data_vars:
                        from sarvalanche.probabilities.pipelines import get_static_probabilities
                        ref_date = pd.Timestamp(ds["time"].values[len(ds["time"]) // 2])
                        ds = get_static_probabilities(ds, ref_date)
                    ds_cache[snc_key] = ds

                ds = ds_cache[snc_key]
                # Clean stale variables
                stale = [v for v in ds.data_vars if re.match(r"^(p|d)_\d+_V[VH]_empirical$", v)
                         or v in ("p_empirical", "d_empirical", "w_temporal")]
                if stale:
                    ds = ds.drop_vars(stale)
                    ds_cache[snc_key] = ds

                log.info("Computing d_empirical for %s (%s)", cnn_date, season_nc.name)
                d_emp = compute_d_empirical_for_date(ds, pd.Timestamp(cnn_date))
                if d_emp is not None:
                    demp_cache[demp_key] = (d_emp.values, d_emp.y.values, d_emp.x.values)
                else:
                    demp_cache[demp_key] = None

            cached = demp_cache[demp_key]
            if cached is not None:
                d_vals, d_y, d_x = cached
                d_emp_clip, d_emp_extent = clip_to_extent(d_vals, d_y, d_x, obs_lng, obs_lat, PLOT_BUFFER_DEG)

        # ── FlowPy path ──────────────────────────────────────────────
        obs_paths = all_paths[all_paths["obs_id"] == obs_id]
        path_geom = obs_paths.geometry.union_all() if len(obs_paths) > 0 else None

        # ── DEM clip for hillshade ─────────────────────────────────────
        dem_clip = None
        if season_nc is not None:
            snc_key = str(season_nc)
            ds = ds_cache.get(snc_key)
            if ds is not None and "dem" in ds.data_vars:
                dem_clip, _ = clip_to_extent(
                    ds["dem"].values, ds.y.values, ds.x.values,
                    obs_lng, obs_lat, PLOT_BUFFER_DEG,
                )

        # ── Feature-based diagnosis ───────────────────────────────────
        feat_stats = {}
        if season_nc is not None and path_geom is not None:
            snc_key = str(season_nc)
            ds = ds_cache.get(snc_key)
            if ds is not None:
                try:
                    feat_stats = get_feature_stats(ds, path_geom, obs_date)
                except Exception as e:
                    log.warning("Feature stats failed for %s: %s", obs_id, e)

            # Add d_empirical stats from demp_cache (computed separately)
            demp_key = (snc_key, obs_date)
            cached_demp = demp_cache.get(demp_key)
            if cached_demp is not None and path_geom is not None:
                try:
                    d_vals, d_y, d_x = cached_demp
                    mask = make_path_mask(path_geom, d_y, d_x)
                    inside = d_vals[mask]
                    inside = inside[np.isfinite(inside)]
                    if len(inside) > 0:
                        feat_stats["d_emp_mean"] = float(np.mean(inside))
                        feat_stats["d_emp_max"] = float(np.max(inside))
                        feat_stats["d_emp_min"] = float(np.min(inside))
                except Exception:
                    pass

        diagnosis = diagnose_miss(row, feat_stats)
        log.info("  Diagnosis: %s", diagnosis)

        obs_aspect = row.get("aspect", "")
        obs_elev = row.get("elevation", np.nan)

        plot_observation(
            d_emp_clip=d_emp_clip, d_emp_extent=d_emp_extent,
            prob_clip=prob_clip, prob_extent=prob_extent,
            dem_clip=dem_clip,
            obs_lng=obs_lng, obs_lat=obs_lat,
            path_geom=path_geom,
            obs_id=obs_id, obs_date=obs_date, cnn_date=cnn_date,
            d_size=d_size, avy_type=avy_type, zone_name=zone_name,
            obs_aspect=obs_aspect, obs_elev=obs_elev,
            max_prob=max_prob, onset_diff=onset_diff, diagnosis=diagnosis,
            out_path=out_path,
        )
        n_plotted += 1

    log.info("Done — %d plots in %s", n_plotted, OUT_DIR)


if __name__ == "__main__":
    main()
