"""Auto-label avalanche debris polygons from d_empirical + terrain heuristics.

Replaces the QGIS manual labeling step. Produces avalanche_labels_<date>.gpkg
files in the same format as human labels, but in a separate output directory
so they never mix with hand-drawn polygons.

The labels are intentionally generous — assign_confidence.py will down-weight
dubious ones via slope, season, cross-ratio, and d_empirical signal strength.

For each date:
  1. Compute d_empirical (backscatter change statistic)
  2. Within each footprint window, threshold d_empirical + terrain filters
  3. Connected-component analysis → candidate polygons
  4. Filter by minimum size and shape
  5. Write to auto_labels_<date>.gpkg (NOT avalanche_labels_ to avoid mixing)

Prerequisite: run pre_v2_cnn_labeling.py first to generate footprint windows.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/auto_label.py \
        --nc local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2024-2025_Turnagain_Pass_and_Girdwood.nc \
        --footprints-dir local/debris_shapes/CNFAIC \
        --out-dir local/debris_shapes/CNFAIC_auto \
        --dates 2025-01-15 2025-02-01

    # Then run post_v2_cnn_labeling.py pointing at --labels-dir local/debris_shapes/CNFAIC_auto
"""

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import box, shape

from sarvalanche.io.dataset import load_netcdf_to_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

TAU = 6

# ── Thresholds ────────────────────────────────────────────────────────
# d_empirical is in dB; positive = increased backscatter (debris signature)
D_EMP_THRESHOLD = 3.0        # minimum dB change to consider
SLOPE_MIN_RAD = 0.09         # ~5° minimum slope (runout zones can be gentle)
SLOPE_MAX_RAD = 1.05         # ~60° maximum slope
MIN_COMPONENT_PX = 4         # minimum connected component size in pixels
MIN_AREA_M2 = 500            # minimum polygon area in m²
MAX_AREA_M2 = 2_000_000      # maximum polygon area (2 km²) — reject huge blobs
MAX_POLYGONS_PER_WINDOW = 500 # safety cap per window; warns if exceeded


# ── Empirical computation (same as pre/post scripts) ─────────────────

def compute_empirical_for_date(ds, reference_date, tau_days=TAU):
    """Compute per-track/pol empirical layers for a given date."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)

    stale_patterns = [
        re.compile(r"^p_\d+_V[VH]_empirical$"),
        re.compile(r"^d_\d+_V[VH]_empirical$"),
        re.compile(r"^m_\d+_V[VH]_empirical$"),
    ]
    stale_exact = {"p_empirical", "d_empirical", "w_temporal"}
    to_drop = [v for v in ds.data_vars if v in stale_exact]
    for pat in stale_patterns:
        to_drop.extend(v for v in ds.data_vars if pat.match(v) and v not in to_drop)
    if to_drop:
        ds = ds.drop_vars(to_drop)

    ds["w_temporal"] = get_temporal_weights(ds["time"], ref_ts, tau_days=tau_days)
    ds["w_temporal"].attrs = {"source": "sarvalanche", "units": "1", "product": "weight"}

    p_empirical, d_empirical = calculate_empirical_backscatter_probability(
        ds, ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical
    return ds


# ── Resolution helper ────────────────────────────────────────────────

def _get_pixel_area_m2(ds):
    """Get approximate pixel area in m²."""
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    crs = ds.rio.crs
    if crs and crs.is_geographic:
        mid_lat = float(ds.y.values.mean())
        dx_m = dx * 111320 * np.cos(np.radians(mid_lat))
        dy_m = dy * 110540
        return dx_m * dy_m
    return dx * dy


# ── Core auto-labeling ───────────────────────────────────────────────

def auto_label_window(ds, y0, y1, x0, x1, pixel_area_m2,
                      d_threshold=D_EMP_THRESHOLD, min_area=MIN_AREA_M2):
    """Generate debris polygons within a window using thresholding + terrain filters.

    Returns a list of shapely polygons (in dataset CRS).
    """
    d_emp = ds["d_empirical"].values[y0:y1, x0:x1]
    slope = ds["slope"].values[y0:y1, x0:x1]

    # Optional terrain filters
    has_cell_counts = "cell_counts" in ds.data_vars
    has_water = "water_mask" in ds.data_vars

    # Build candidate mask
    candidates = (
        (d_emp >= d_threshold)
        & (slope >= SLOPE_MIN_RAD)
        & (slope <= SLOPE_MAX_RAD)
        & np.isfinite(d_emp)
    )

    # Exclude water
    if has_water:
        water = ds["water_mask"].values[y0:y1, x0:x1]
        candidates = candidates & (water == 0)

    # Boost: if FlowPy says runout is possible here, lower the d_emp threshold
    if has_cell_counts:
        cell_counts = ds["cell_counts"].values[y0:y1, x0:x1]
        runout_boost = (
            (d_emp >= d_threshold * 0.75)
            & (cell_counts > 0)
            & (slope >= SLOPE_MIN_RAD)
            & (slope <= SLOPE_MAX_RAD)
            & np.isfinite(d_emp)
        )
        if has_water:
            runout_boost = runout_boost & (water == 0)
        candidates = candidates | runout_boost

    # Connected component labeling
    labeled, n_components = ndimage.label(candidates)
    if n_components == 0:
        return []

    # Filter by size
    component_sizes = ndimage.sum(candidates, labeled, range(1, n_components + 1))

    # Build rasterio transform for this window
    x_vals = ds.x.values[x0:x1]
    y_vals = ds.y.values[y0:y1]
    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    dy = abs(float(ds.y.values[1] - ds.y.values[0]))
    transform = from_bounds(
        float(x_vals.min()) - dx / 2, float(y_vals.min()) - dy / 2,
        float(x_vals.max()) + dx / 2, float(y_vals.max()) + dy / 2,
        len(x_vals), len(y_vals),
    )

    polygons = []
    for comp_id in range(1, n_components + 1):
        size_px = component_sizes[comp_id - 1]
        if size_px < MIN_COMPONENT_PX:
            continue

        area_m2 = size_px * pixel_area_m2
        if area_m2 < min_area or area_m2 > MAX_AREA_M2:
            continue

        # Mean d_empirical in this component
        comp_mask = labeled == comp_id
        mean_d = float(np.nanmean(d_emp[comp_mask]))

        # Convert to polygon
        comp_binary = (labeled == comp_id).astype(np.uint8)
        for geom, val in rasterio_shapes(comp_binary, transform=transform):
            if val == 1:
                poly = shape(geom)
                if poly.is_valid and not poly.is_empty:
                    polygons.append({
                        "geometry": poly,
                        "mean_d_empirical": round(mean_d, 2),
                        "area_m2": round(area_m2, 0),
                        "n_pixels": int(size_px),
                        "source": "auto",
                    })

    if len(polygons) > MAX_POLYGONS_PER_WINDOW:
        log.warning(
            "  %d polygons exceeds cap (%d) — keeping top %d by mean_d_empirical. "
            "Consider raising --d-threshold.",
            len(polygons), MAX_POLYGONS_PER_WINDOW, MAX_POLYGONS_PER_WINDOW,
        )
        polygons.sort(key=lambda p: p["mean_d_empirical"], reverse=True)
        polygons = polygons[:MAX_POLYGONS_PER_WINDOW]

    return polygons


def auto_label_date(ds, date_str, footprints_dir, out_dir, tau,
                    d_threshold=D_EMP_THRESHOLD, min_area=MIN_AREA_M2):
    """Auto-label a single date. Returns number of polygons generated."""
    ref_date = pd.Timestamp(date_str)
    ds = compute_empirical_for_date(ds, ref_date, tau)

    pixel_area = _get_pixel_area_m2(ds)
    log.info("  Pixel area: %.0f m²", pixel_area)

    # Determine windows from geotiffs or use full scene
    geotiff_dir = footprints_dir / "geotiffs" / date_str
    if geotiff_dir.is_dir() and list(geotiff_dir.glob("*.tif")):
        import rasterio
        windows = []
        for tif in sorted(geotiff_dir.glob("*.tif")):
            with rasterio.open(tif) as src:
                b = src.bounds
            fp = box(b.left, b.bottom, b.right, b.top)
            # Convert to pixel coords
            minx, miny, maxx, maxy = fp.bounds
            x = ds.x.values
            y = ds.y.values
            x_start = int(np.searchsorted(x, minx))
            x_end = int(np.searchsorted(x, maxx))
            if y[0] > y[-1]:
                y_start = int(np.searchsorted(-y, -maxy))
                y_end = int(np.searchsorted(-y, -miny))
            else:
                y_start = int(np.searchsorted(y, miny))
                y_end = int(np.searchsorted(y, maxy))
            windows.append((y_start, y_end, x_start, x_end))
            log.info("  Window from %s: y[%d:%d] x[%d:%d]", tif.name, y_start, y_end, x_start, x_end)
    else:
        log.info("  No geotiffs found for %s — using full scene", date_str)
        H, W = ds.sizes["y"], ds.sizes["x"]
        windows = [(0, H, 0, W)]

    # Generate labels for each window
    all_polys = []
    for y0, y1, x0, x1 in windows:
        polys = auto_label_window(ds, y0, y1, x0, x1, pixel_area,
                                  d_threshold=d_threshold, min_area=min_area)
        all_polys.extend(polys)
        log.info("  Window y[%d:%d] x[%d:%d]: %d polygons", y0, y1, x0, x1, len(polys))

    if not all_polys:
        log.warning("  No debris candidates found for %s", date_str)
        return 0

    # Write GeoPackage
    gdf = gpd.GeoDataFrame(all_polys, crs=ds.rio.crs)
    out_path = out_dir / f"avalanche_labels_{date_str}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    log.info(
        "  Wrote %d polygons to %s (total area: %.0f m²)",
        len(gdf), out_path.name, gdf["area_m2"].sum(),
    )

    # Also copy footprint data so post_v2_cnn_labeling can find geotiffs
    auto_geotiff_dir = out_dir / "geotiffs" / date_str
    if geotiff_dir.is_dir() and not auto_geotiff_dir.exists():
        auto_geotiff_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for tif in geotiff_dir.glob("*.tif"):
            shutil.copy2(tif, auto_geotiff_dir / tif.name)
        log.info("  Copied %d geotiffs to %s", len(list(geotiff_dir.glob("*.tif"))), auto_geotiff_dir)

    return len(gdf)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label debris polygons from d_empirical + terrain heuristics",
    )
    parser.add_argument(
        "--nc", type=Path, required=True,
        help="Path to season_dataset.nc",
    )
    parser.add_argument(
        "--footprints-dir", type=Path, required=True,
        help="Directory with geotiffs/<date>/ from pre_v2_cnn_labeling.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for auto-generated labels (separate from human labels!)",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Specific dates to label (default: all dates with geotiffs)",
    )
    parser.add_argument(
        "--tau", type=float, default=TAU,
        help=f"Temporal decay tau in days (default: {TAU})",
    )
    parser.add_argument(
        "--d-threshold", type=float, default=D_EMP_THRESHOLD,
        help=f"Minimum d_empirical threshold in dB (default: {D_EMP_THRESHOLD})",
    )
    parser.add_argument(
        "--min-area", type=float, default=MIN_AREA_M2,
        help=f"Minimum polygon area in m² (default: {MIN_AREA_M2})",
    )
    args = parser.parse_args()

    # Discover dates
    if args.dates:
        dates = args.dates
    else:
        geotiff_root = args.footprints_dir / "geotiffs"
        if geotiff_root.is_dir():
            dates = sorted(
                d.name for d in geotiff_root.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            )
        else:
            parser.error("No --dates provided and no geotiffs/ directory found")

    log.info("Auto-labeling %d dates", len(dates))

    # Load dataset once
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  %d time steps, %d×%d spatial", len(ds.time), ds.sizes["y"], ds.sizes["x"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process each date
    total_polys = 0
    for date_str in dates:
        log.info("=== Date: %s ===", date_str)
        n = auto_label_date(ds, date_str, args.footprints_dir, args.out_dir, args.tau,
                            d_threshold=args.d_threshold, min_area=args.min_area)
        total_polys += n

    log.info("")
    log.info("Done. %d total polygons across %d dates", total_polys, len(dates))
    log.info("Output: %s", args.out_dir)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Optionally review auto-labels in QGIS")
    log.info("  2. Run post_v2_cnn_labeling.py with --labels-dir %s", args.out_dir)


if __name__ == "__main__":
    main()
