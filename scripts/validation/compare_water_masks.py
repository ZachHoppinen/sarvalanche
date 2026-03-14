"""
Compare NLCD vs ESA WorldCover vs JRC Global Surface Water masks over a CONUS area.

Uses a region in the Wasatch Range (Utah) that has both mountains and
reservoirs/lakes, so we can see how the three datasets agree on water bodies
in avalanche-relevant terrain.
"""

import logging
import math

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
import pygeohydro as gh
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import box

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── AOI: Little Cottonwood Canyon / Alta area, Utah ──────────────────
# Includes Tibble Fork Reservoir, Silver Lake, and surrounding mountains
# Bounding box in EPSG:4326 (lon_min, lat_min, lon_max, lat_max)
AOI_BOUNDS = (-111.70, 40.52, -111.55, 40.62)
AOI_CRS = "EPSG:4326"

# JRC occurrence threshold: pixels with water occurrence >= this % are water
JRC_OCCURRENCE_THRESHOLD = 50


# ── NLCD ─────────────────────────────────────────────────────────────

def get_nlcd_water(aoi, aoi_crs):
    """Fetch NLCD water mask (class 11 = Open Water)."""
    g = gpd.GeoSeries([aoi], crs=aoi_crs)
    years = {'impervious': [2021], 'cover': [2021], 'canopy': [2021], 'descriptor': [2021]}
    lc = gh.nlcd_bygeom(geometry=g, years=years)[0]["cover_2021"]
    if "band" in lc.dims:
        lc = lc.isel(band=0, drop=True)
    water = (lc == 11).astype(np.uint8)
    water.attrs = {"source": "nlcd", "description": "1=water"}
    return water, lc


# ── ESA WorldCover ───────────────────────────────────────────────────

def _esa_tile_url(lat, lon):
    """Build ESA WorldCover 2021 v200 tile URL for a given lat/lon corner.

    Tiles are 3x3 degree, named by their SW corner.
    Example: N39W114 covers 39-42N, 114-111W.
    """
    lat_tile = int(math.floor(lat / 3) * 3)
    lon_tile = int(math.floor(lon / 3) * 3)
    lat_str = f"N{abs(lat_tile):02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
    lon_str = f"E{abs(lon_tile):03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
    return (
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
    )


def get_esa_water(aoi, aoi_crs):
    """Fetch ESA WorldCover and extract water class (80)."""
    bounds = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds

    urls = set()
    for lat in [miny, maxy]:
        for lon in [minx, maxx]:
            urls.add(_esa_tile_url(lat, lon))

    log.info("ESA WorldCover tiles: %s", urls)

    pieces = []
    for url in urls:
        try:
            da = rioxarray.open_rasterio(url)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            if "band" in clipped.dims:
                clipped = clipped.isel(band=0, drop=True)
            pieces.append(clipped)
        except Exception as e:
            log.warning("ESA tile %s failed: %s", url, e)

    if not pieces:
        raise RuntimeError("No ESA WorldCover tiles loaded")

    if len(pieces) == 1:
        lc = pieces[0]
    else:
        from rioxarray.merge import merge_arrays
        lc = merge_arrays(pieces)

    water = (lc == 80).astype(np.uint8)
    water.attrs = {"source": "esa_worldcover", "description": "1=water (class 80)"}
    return water, lc


# ── JRC Global Surface Water ────────────────────────────────────────

def _jrc_tile_url(lat, lon):
    """Build JRC Global Surface Water occurrence tile URL.

    Tiles are 10x10 degree. Naming uses the tile corner:
    occurrence_{abs_lon}{E/W}_{abs_lat}{N/S}v1_4_2021.tif
    """
    # Tile corners: round down to nearest 10 for positive, round towards
    # more-negative for negative
    lat_tile = int(math.ceil(lat / 10) * 10)  # upper edge
    lon_tile = int(math.ceil(abs(lon) / 10) * 10)  # round up abs value

    lat_str = f"{abs(lat_tile)}{('N' if lat_tile >= 0 else 'S')}"
    lon_str = f"{abs(lon_tile)}{('E' if lon >= 0 else 'W')}"

    return (
        f"https://storage.googleapis.com/global-surface-water/downloads2021/"
        f"occurrence/occurrence_{lon_str}_{lat_str}v1_4_2021.tif"
    )


def get_jrc_water(aoi, aoi_crs, threshold=JRC_OCCURRENCE_THRESHOLD):
    """Fetch JRC Global Surface Water occurrence and threshold to binary mask.

    Parameters
    ----------
    threshold : int
        Minimum occurrence percentage (0-100) to classify as water.
        Default 50 means pixel is water if it was water >= 50% of observations.
    """
    bounds = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds

    urls = set()
    for lat in [miny, maxy]:
        for lon in [minx, maxx]:
            urls.add(_jrc_tile_url(lat, lon))

    log.info("JRC GSW tiles: %s", urls)

    pieces = []
    for url in urls:
        try:
            da = rioxarray.open_rasterio(url)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            if "band" in clipped.dims:
                clipped = clipped.isel(band=0, drop=True)
            pieces.append(clipped.astype(np.float32))
        except Exception as e:
            log.warning("JRC tile %s failed: %s", url, e)

    if not pieces:
        raise RuntimeError("No JRC Global Surface Water tiles loaded")

    if len(pieces) == 1:
        occurrence = pieces[0]
    else:
        from rioxarray.merge import merge_arrays
        occurrence = merge_arrays(pieces)

    # Occurrence is 0-100 (% of valid observations where water was detected)
    # Values 255 = no data, 0 = never water
    water = ((occurrence >= threshold) & (occurrence <= 100)).astype(np.uint8)
    water.attrs = {
        "source": "jrc_gsw",
        "description": f"1=water (occurrence >= {threshold}%)",
    }
    return water, occurrence


# ── Comparison ───────────────────────────────────────────────────────

def compare_and_plot(nlcd_water, esa_water, jrc_water, nlcd_lc, esa_lc, jrc_occ):
    """Reproject ESA and JRC to NLCD grid and compare all three."""
    esa_matched = esa_water.rio.reproject_match(nlcd_water, resampling=0)
    jrc_matched = jrc_water.rio.reproject_match(nlcd_water, resampling=0)

    nlcd_w = nlcd_water.values.astype(bool)
    esa_w = esa_matched.values.astype(bool)
    jrc_w = jrc_matched.values.astype(bool)
    total = nlcd_w.size

    # Pairwise stats
    def _stats(name_a, a, name_b, b):
        both = a & b
        a_only = a & ~b
        b_only = ~a & b
        union = a | b
        iou = both.sum() / union.sum() if union.sum() > 0 else float("nan")
        log.info(f"  {name_a} vs {name_b}:")
        log.info(f"    Both:       {both.sum():>7d}  ({both.sum()/total*100:.2f}%)")
        log.info(f"    {name_a} only: {a_only.sum():>7d}  ({a_only.sum()/total*100:.2f}%)")
        log.info(f"    {name_b} only: {b_only.sum():>7d}  ({b_only.sum()/total*100:.2f}%)")
        log.info(f"    IoU:        {iou:.3f}")
        return iou

    log.info("=== Water mask comparison (at NLCD 30m resolution) ===")
    log.info(f"  NLCD water pixels:  {nlcd_w.sum()}")
    log.info(f"  ESA water pixels:   {esa_w.sum()}")
    log.info(f"  JRC water pixels:   {jrc_w.sum()}")
    log.info("")
    iou_nlcd_esa = _stats("NLCD", nlcd_w, "ESA", esa_w)
    log.info("")
    iou_nlcd_jrc = _stats("NLCD", nlcd_w, "JRC", jrc_w)
    log.info("")
    iou_esa_jrc = _stats("ESA", esa_w, "JRC", jrc_w)

    # All-three agreement
    all_three = nlcd_w & esa_w & jrc_w
    any_one = nlcd_w | esa_w | jrc_w
    log.info("")
    log.info(f"  All three agree water: {all_three.sum()}")
    log.info(f"  Any one says water:    {any_one.sum()}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Individual masks
    axes[0, 0].imshow(nlcd_water.values, cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set_title(f"NLCD Water (class 11)\n{nlcd_w.sum()} px")

    axes[0, 1].imshow(esa_matched.values, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title(f"ESA WorldCover Water (class 80)\n{esa_w.sum()} px")

    axes[0, 2].imshow(jrc_matched.values, cmap="Blues", vmin=0, vmax=1)
    axes[0, 2].set_title(f"JRC GSW (occ >= {JRC_OCCURRENCE_THRESHOLD}%)\n{jrc_w.sum()} px")

    # Row 2: Pairwise agreements
    def _agreement_map(a, b, ax, title, name_a, name_b, iou):
        both = a & b
        a_only = a & ~b
        b_only = ~a & b
        agree = np.zeros_like(a, dtype=np.uint8)
        agree[both] = 1
        agree[a_only] = 2
        agree[b_only] = 3
        cmap = ListedColormap(["#f0f0f0", "#2166ac", "#d6604d", "#4dac26"])
        im = ax.imshow(agree, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"{name_a} vs {name_b}\nIoU={iou:.3f}")
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
        cbar.ax.set_yticklabels(["Land", "Both", f"{name_a} only", f"{name_b} only"])

    _agreement_map(nlcd_w, esa_w, axes[1, 0], "", "NLCD", "ESA", iou_nlcd_esa)
    _agreement_map(nlcd_w, jrc_w, axes[1, 1], "", "NLCD", "JRC", iou_nlcd_jrc)
    _agreement_map(esa_w, jrc_w, axes[1, 2], "", "ESA", "JRC", iou_esa_jrc)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        f"Water mask comparison – Little Cottonwood Canyon, UT\n"
        f"NLCD={nlcd_w.sum()} px | ESA={esa_w.sum()} px | JRC={jrc_w.sum()} px",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = "scripts/validation/water_mask_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved figure to {out_path}")
    plt.show()


def main():
    aoi = box(*AOI_BOUNDS)

    log.info("Fetching NLCD water mask...")
    nlcd_water, nlcd_lc = get_nlcd_water(aoi, AOI_CRS)
    log.info(f"  NLCD shape: {nlcd_water.shape}, water pixels: {(nlcd_water==1).sum().values}")

    log.info("Fetching ESA WorldCover water mask...")
    esa_water, esa_lc = get_esa_water(aoi, AOI_CRS)
    log.info(f"  ESA shape: {esa_water.shape}, water pixels: {(esa_water==1).sum().values}")

    log.info("Fetching JRC Global Surface Water...")
    jrc_water, jrc_occ = get_jrc_water(aoi, AOI_CRS)
    log.info(f"  JRC shape: {jrc_water.shape}, water pixels: {(jrc_water==1).sum().values}")

    compare_and_plot(nlcd_water, esa_water, jrc_water, nlcd_lc, esa_lc, jrc_occ)


if __name__ == "__main__":
    main()
