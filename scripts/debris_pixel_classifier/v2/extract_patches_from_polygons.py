"""Extract v2 training patches from manually drawn debris polygons.

Takes a GeoPackage of debris polygons, a season dataset, and optionally
GeoTIFF footprints defining the reviewed area. Patches are only extracted
within footprint windows so that areas outside (where you haven't looked)
don't pollute training with false negatives.

Sampling strategy:
  - Within each footprint window, slide at --stride extracting 128×128 patches
  - Patches overlapping debris polygons → positive (pixel-level mask)
  - Patches with no debris overlap → negative
  - --neg-ratio controls max negative:positive ratio

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/extract_patches_from_polygons.py \
        --nc 'local/issw/dual_tau_output/Sawtooth_&_Western_Smoky_Mtns/season_dataset.nc' \
        --polygons local/issw/debris_shapes/avalanche_labels_2025-02-04.gpkg \
        --footprints local/issw/debris_shapes/patch_footprints.gpkg \
        --date 2025-02-04 \
        --tau 6 \
        --out-dir local/issw/v2_patches/2025-02-04
"""

import argparse
import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import box

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v2.patch_extraction import (
    V2_PATCH_SIZE,
    build_v2_patch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def compute_empirical_for_date(ds, reference_date, tau_days):
    """Compute per-track/pol empirical layers for a given date."""
    from sarvalanche.detection.backscatter_change import (
        calculate_empirical_backscatter_probability,
    )
    from sarvalanche.weights.temporal import get_temporal_weights

    ref_ts = np.datetime64(reference_date)

    stale_patterns = [
        re.compile(r"^p_\d+_V[VH]_empirical$"),
        re.compile(r"^d_\d+_V[VH]_empirical$"),
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
        ds,
        ref_ts,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )
    ds["p_empirical"] = p_empirical
    ds["d_empirical"] = d_empirical
    return ds


def rasterize_polygons(gdf, ds):
    """Rasterize polygons to a binary (H, W) mask aligned with dataset grid."""
    x = ds.x.values
    y = ds.y.values
    H, W = len(y), len(x)

    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))

    transform = from_bounds(
        float(x.min()) - dx / 2, float(y.min()) - dy / 2,
        float(x.max()) + dx / 2, float(y.max()) + dy / 2,
        W, H,
    )

    ds_crs = ds.rio.crs
    if ds_crs and gdf.crs and gdf.crs != ds_crs:
        gdf = gdf.to_crs(ds_crs)

    mask = ~geometry_mask(
        gdf.geometry,
        out_shape=(H, W),
        transform=transform,
        all_touched=True,
    )
    return mask.astype(np.float32)


def footprints_from_geotiffs(geotiff_dir):
    """Build a GeoDataFrame of footprint polygons from all .tif files in a directory."""
    tif_paths = sorted(Path(geotiff_dir).glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {geotiff_dir}")

    rows = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            rows.append({
                "geometry": box(b.left, b.bottom, b.right, b.top),
                "geotiff": p.name,
            })
        crs = src.crs

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    log.info("Built %d footprints from GeoTIFFs in %s", len(gdf), geotiff_dir)
    return gdf


def footprint_to_pixel_bounds(footprint_geom, ds):
    """Convert a footprint polygon to (y_start, y_end, x_start, x_end) pixel indices."""
    minx, miny, maxx, maxy = footprint_geom.bounds
    x = ds.x.values
    y = ds.y.values

    x_start = int(np.searchsorted(x, minx))
    x_end = int(np.searchsorted(x, maxx))
    # y may be descending
    if y[0] > y[-1]:
        y_start = int(np.searchsorted(-y, -maxy))
        y_end = int(np.searchsorted(-y, -miny))
    else:
        y_start = int(np.searchsorted(y, miny))
        y_end = int(np.searchsorted(y, maxy))

    return y_start, y_end, x_start, x_end


def extract_patches(
    ds,
    debris_mask,
    footprint_windows,
    out_dir,
    patch_size=V2_PATCH_SIZE,
    stride=64,
    neg_ratio=3.0,
    min_debris_frac=0.005,
):
    """Extract positive and negative v2 patches within footprint windows.

    Parameters
    ----------
    ds : xr.Dataset
    debris_mask : (H, W) float32 binary mask
    footprint_windows : list of (y_start, y_end, x_start, x_end) tuples
    out_dir : Path
    patch_size : int
    stride : int
    neg_ratio : float
        Max ratio of negative to positive patches.
    min_debris_frac : float
        Min fraction of debris pixels for a patch to count as positive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_patches = []
    neg_patches = []

    for y_start, y_end, x_start, x_end in footprint_windows:
        for y0 in range(y_start, y_end - patch_size + 1, stride):
            for x0 in range(x_start, x_end - patch_size + 1, stride):
                patch_mask = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size]
                debris_frac = patch_mask.mean()
                if debris_frac >= min_debris_frac:
                    pos_patches.append((y0, x0, debris_frac))
                else:
                    neg_patches.append((y0, x0, 0.0))

    log.info("Found %d positive, %d negative candidate patches", len(pos_patches), len(neg_patches))

    # Subsample negatives
    n_neg_max = int(len(pos_patches) * neg_ratio)
    if len(neg_patches) > n_neg_max:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(neg_patches), size=n_neg_max, replace=False)
        neg_patches = [neg_patches[i] for i in sorted(indices)]
        log.info("Subsampled to %d negatives (%.1f:1 ratio)", len(neg_patches), neg_ratio)

    # Save all patches
    metadata = {}
    n_saved = 0

    for label, patches in [(1, pos_patches), (0, neg_patches)]:
        for y0, x0, debris_frac in patches:
            sar_maps, static = build_v2_patch(ds, y0, x0, patch_size)

            # Skip patches with no SAR signal
            if np.abs(sar_maps).max() < 1e-6:
                continue

            # Pixel-level label from debris mask
            patch_label = debris_mask[y0:y0 + patch_size, x0:x0 + patch_size]

            tile_id = f"{'pos' if label == 1 else 'neg'}_{y0:04d}_{x0:04d}"
            np.savez_compressed(
                out_dir / f"{tile_id}_v2_.npz",
                sar_maps=sar_maps,
                static=static,
                label=np.int8(label),
                label_mask=patch_label,
                x_coords=ds.x.values[x0:x0 + patch_size],
                y_coords=ds.y.values[y0:y0 + patch_size],
                crs=str(ds.rio.crs),
            )
            metadata[tile_id] = {
                'label': int(label),
                'y0': int(y0),
                'x0': int(x0),
                'debris_frac': float(debris_frac),
            }
            n_saved += 1

    # Save metadata
    labels_path = out_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    n_pos = sum(1 for v in metadata.values() if v['label'] == 1)
    n_neg = sum(1 for v in metadata.values() if v['label'] == 0)
    log.info(
        "Saved %d patches (%d pos, %d neg) to %s",
        n_saved, n_pos, n_neg, out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract v2 training patches from debris polygon labels"
    )
    parser.add_argument("--nc", type=Path, required=True, help="Path to season_dataset.nc")
    parser.add_argument("--polygons", type=Path, required=True, help="GeoPackage with debris polygons")
    parser.add_argument("--footprints", type=Path, default=None,
                        help="GeoPackage with reviewed window footprints (limits extraction area)")
    parser.add_argument("--geotiff-dir", type=Path, default=None,
                        help="Directory of GeoTIFFs to derive footprints from (overrides --footprints)")
    parser.add_argument("--date", type=str, required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--tau", type=float, default=6, help="Temporal decay tau (default: 6)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for patches")
    parser.add_argument("--stride", type=int, default=64,
                        help="Stride for patch extraction (default: 64)")
    parser.add_argument("--neg-ratio", type=float, default=3.0,
                        help="Max negative:positive patch ratio (default: 3.0)")
    parser.add_argument("--min-debris-frac", type=float, default=0.005,
                        help="Min debris fraction for positive (default: 0.005)")
    args = parser.parse_args()

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()
    log.info("  %d time steps, %d×%d spatial", len(ds.time), ds.sizes['y'], ds.sizes['x'])

    # Ensure w_resolution
    if "w_resolution" not in ds.data_vars:
        raise RuntimeError("w_resolution not found — re-run season pipeline first")

    # Compute empirical
    ref_date = pd.Timestamp(args.date)
    log.info("Computing empirical for %s (tau=%gd)...", args.date, args.tau)
    ds = compute_empirical_for_date(ds, ref_date, args.tau)

    # Load and rasterize debris polygons
    log.info("Loading polygons from %s", args.polygons)
    gdf = gpd.read_file(args.polygons)
    log.info("  %d polygons", len(gdf))

    log.info("Rasterizing polygons to dataset grid...")
    debris_mask = rasterize_polygons(gdf, ds)
    log.info(
        "  Debris pixels: %d / %d (%.2f%%)",
        int(debris_mask.sum()),
        debris_mask.size,
        100 * debris_mask.mean(),
    )

    # Determine extraction windows
    if args.geotiff_dir:
        fp_gdf = footprints_from_geotiffs(args.geotiff_dir)
        ds_crs = ds.rio.crs
        if ds_crs and fp_gdf.crs and fp_gdf.crs != ds_crs:
            fp_gdf = fp_gdf.to_crs(ds_crs)

        footprint_windows = []
        for _, row in fp_gdf.iterrows():
            bounds = footprint_to_pixel_bounds(row.geometry, ds)
            footprint_windows.append(bounds)
            y_start, y_end, x_start, x_end = bounds
            log.info("    Window: y[%d:%d] x[%d:%d] (%s)", y_start, y_end, x_start, x_end, row["geotiff"])
    elif args.footprints:
        log.info("Loading footprints from %s", args.footprints)
        fp_gdf = gpd.read_file(args.footprints)
        if 'date' in fp_gdf.columns:
            fp_gdf = fp_gdf[fp_gdf['date'] == args.date]
        log.info("  %d footprint windows for %s", len(fp_gdf), args.date)

        footprint_windows = []
        for _, row in fp_gdf.iterrows():
            bounds = footprint_to_pixel_bounds(row.geometry, ds)
            footprint_windows.append(bounds)
            y_start, y_end, x_start, x_end = bounds
            log.info("    Window: y[%d:%d] x[%d:%d]", y_start, y_end, x_start, x_end)
    else:
        # Use full scene
        H, W = ds.sizes['y'], ds.sizes['x']
        footprint_windows = [(0, H, 0, W)]
        log.info("No footprints provided — using full scene")

    # Extract patches
    extract_patches(
        ds,
        debris_mask,
        footprint_windows,
        args.out_dir,
        stride=args.stride,
        neg_ratio=args.neg_ratio,
        min_debris_frac=args.min_debris_frac,
    )


if __name__ == "__main__":
    main()
