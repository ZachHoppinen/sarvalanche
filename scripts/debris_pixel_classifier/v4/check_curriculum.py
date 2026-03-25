"""Check if a specific label shape passes the curriculum filter for a given pair.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v4/check_curriculum.py \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --polygons local/issw/debris_shapes/snfac/avalanche_labels_2025-02-19.gpkg \
        --track 71 --t-start 2025-02-14 --t-end 2025-02-26 \
        --shape-id 81

    # Check all shapes:
        --shape-id all

    # Check a range:
        --shape-id 80-90
"""

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features
from rasterio.transform import from_bounds
from pathlib import Path

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.patch_extraction import get_all_season_pairs, V3_PATCH_SIZE


def check_shape(change_vv, debris_mask, shape_mask, shape_id, stride=64):
    """Check curriculum filter for one shape. Returns dict of results."""
    ps = V3_PATCH_SIZE
    H, W = change_vv.shape

    # Shape-level signal
    has_cov = np.abs(change_vv[shape_mask > 0.5]) > 1e-6
    dc = change_vv[shape_mask > 0.5][has_cov]
    n_px = int((shape_mask > 0.5).sum())
    n_cov = int(has_cov.sum())

    if n_cov == 0:
        return {"id": shape_id, "n_px": n_px, "n_cov": 0, "included": "no_coverage"}

    p90 = float(np.percentile(dc, 90))
    p98 = float(np.percentile(dc, 98))
    mn = float(dc.mean())
    contrast = p90 - mn

    raw_mean = float(np.sign(mn) * np.expm1(abs(mn)))
    raw_p90 = float(np.sign(p90) * np.expm1(abs(p90)))
    raw_p98 = float(np.sign(p98) * np.expm1(abs(p98)))

    # Per-patch: check the patch most centered on this shape
    ys, xs = np.where(shape_mask > 0.5)
    cy, cx = int(ys.mean()), int(xs.mean())
    best_y0 = max(0, min(cy - ps // 2, H - ps))
    best_x0 = max(0, min(cx - ps // 2, W - ps))

    # Use FULL debris mask for the patch (matching training)
    patch_label = debris_mask[best_y0:best_y0+ps, best_x0:best_x0+ps]
    patch_cov = np.abs(change_vv[best_y0:best_y0+ps, best_x0:best_x0+ps]) > 1e-6
    patch_debris = (patch_label > 0.5) & patch_cov
    pdc = change_vv[best_y0:best_y0+ps, best_x0:best_x0+ps][patch_debris]

    if len(pdc) > 0:
        patch_p90 = float(np.percentile(pdc, 90))
        patch_p98 = float(np.percentile(pdc, 98))
        patch_mn = float(pdc.mean())
        patch_contrast = patch_p90 - patch_mn
        patch_strict = patch_p90 > 1.0 and patch_contrast > 0.3
        patch_medium = patch_p90 > 0.5
        patch_relaxed = patch_p98 > 0.3
    else:
        patch_strict = patch_medium = patch_relaxed = False

    return {
        "id": shape_id,
        "n_px": n_px,
        "n_cov": n_cov,
        "mean_dB": raw_mean,
        "p90_dB": raw_p90,
        "p98_dB": raw_p98,
        "shape_strict": p90 > 1.0 and contrast > 0.3,
        "shape_medium": p90 > 0.5,
        "shape_relaxed": p98 > 0.3,
        "patch_strict": patch_strict,
        "patch_medium": patch_medium,
        "patch_relaxed": patch_relaxed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--t-start", type=str, required=True)
    parser.add_argument("--t-end", type=str, required=True)
    parser.add_argument("--shape-id", type=str, default="all",
                        help="Shape index: integer, 'all', or 'start-end' range")
    parser.add_argument("--max-span-days", type=int, default=60)
    args = parser.parse_args()

    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    H, W = ds.sizes["y"], ds.sizes["x"]
    x_arr, y_arr = ds.x.values, ds.y.values
    dx = abs(float(x_arr[1] - x_arr[0])); dy = abs(float(y_arr[1] - y_arr[0]))
    transform = from_bounds(
        float(x_arr.min()) - dx/2, float(y_arr.min()) - dy/2,
        float(x_arr.max()) + dx/2, float(y_arr.max()) + dy/2, W, H,
    )

    # Find pair
    pairs = get_all_season_pairs(ds, max_span_days=args.max_span_days)
    target = None
    for p in pairs:
        if (str(p["t_start"])[:10] == args.t_start and
            str(p["t_end"])[:10] == args.t_end and
            str(p["track"]) == str(args.track)):
            target = p; break
    if target is None:
        print(f"Pair not found: trk{args.track} {args.t_start}→{args.t_end}")
        return
    change_vv = target["sar"][0]
    print(f"Pair: trk{args.track} {args.t_start}→{args.t_end} ({target['span_days']}d)")

    # Load polygons
    gdf = gpd.read_file(args.polygons)
    if gdf.crs and ds.rio.crs and gdf.crs != ds.rio.crs:
        gdf = gdf.to_crs(ds.rio.crs)

    # Full debris mask
    debris_mask = rasterio.features.geometry_mask(
        gdf.geometry, out_shape=(H, W), transform=transform,
        invert=True, all_touched=True,
    ).astype(np.float32)

    # Parse shape IDs
    if args.shape_id == "all":
        ids = list(range(len(gdf)))
    elif "-" in args.shape_id:
        s, e = args.shape_id.split("-")
        ids = list(range(int(s), int(e) + 1))
    else:
        ids = [int(args.shape_id)]

    # Header
    print(f"{'ID':>4} {'px':>5} {'cov':>5} {'mean_dB':>8} {'p90_dB':>8} {'p98_dB':>8}  "
          f"{'strict':>6} {'medium':>6} {'relaxed':>7}  "
          f"{'p_strict':>8} {'p_medium':>8} {'p_relaxed':>9}")
    print("-" * 110)

    for sid in ids:
        if sid >= len(gdf):
            continue
        geom = gdf.iloc[sid].geometry
        if geom.is_empty:
            continue
        shape_mask = rasterio.features.geometry_mask(
            [geom], out_shape=(H, W), transform=transform,
            invert=True, all_touched=True,
        ).astype(np.float32)
        if shape_mask.sum() == 0:
            continue

        r = check_shape(change_vv, debris_mask, shape_mask, sid)
        if r.get("included") == "no_coverage":
            print(f"{r['id']:>4} {r['n_px']:>5} {0:>5}  {'--- no coverage ---'}")
            continue

        print(f"{r['id']:>4} {r['n_px']:>5} {r['n_cov']:>5} {r['mean_dB']:>+8.2f} {r['p90_dB']:>+8.2f} {r['p98_dB']:>+8.2f}  "
              f"{'Y' if r['shape_strict'] else '.':>6} {'Y' if r['shape_medium'] else '.':>6} {'Y' if r['shape_relaxed'] else '.':>7}  "
              f"{'Y' if r['patch_strict'] else '.':>8} {'Y' if r['patch_medium'] else '.':>8} {'Y' if r['patch_relaxed'] else '.':>9}")

    ds.close()


if __name__ == "__main__":
    main()
