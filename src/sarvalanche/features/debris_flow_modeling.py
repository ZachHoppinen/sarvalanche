import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation, binary_closing, label as ndlabel

from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

log = logging.getLogger(__name__)


def _apply_linear_split(mask_arr, field_arr, threshold):
    """
    Zero out pixels on both sides of edges where ``field_arr`` changes
    sharply (> ``threshold``), then heal single-pixel holes with a 2×2
    binary closing.

    Parameters
    ----------
    mask_arr : np.ndarray of bool
        Current release-zone binary mask.
    field_arr : np.ndarray of float
        Scalar terrain field (e.g. TPI or curvature).  NaN values are
        treated as zero for differencing.
    threshold : float
        Minimum absolute difference between adjacent pixels to trigger a
        split.

    Returns
    -------
    np.ndarray of bool
    """
    f = np.where(np.isfinite(field_arr), field_arr, 0.0)

    diff_h = np.abs(f[:, :-1] - f[:, 1:])
    diff_v = np.abs(f[:-1, :] - f[1:, :])

    split_mask = binary_dilation(
        (
            np.pad(diff_h > threshold, ((0, 0), (0, 1)), constant_values=False) |
            np.pad(diff_h > threshold, ((0, 0), (1, 0)), constant_values=False) |
            np.pad(diff_v > threshold, ((0, 1), (0, 0)), constant_values=False) |
            np.pad(diff_v > threshold, ((1, 0), (0, 0)), constant_values=False)
        ),
        structure=np.ones((3, 3), dtype=bool),
    )

    result = mask_arr.copy()
    result[split_mask] = False
    return binary_closing(result, structure=np.ones((2, 2), dtype=bool))

def generate_runcount_alpha_angle(ds):
    from sarvalanche.utils.terrain import (
        compute_tpi, compute_curvature, compute_flow_accumulation,
        multiscale_ridgeline_tpi,
    )

    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())
    log.debug("generate_runcount_alpha_angle: DEM shape=%s", dem_proj.shape)

    tpi        = compute_tpi(dem_proj, radius_m=300.0)
    curv       = compute_curvature(dem_proj)
    flow_accum = compute_flow_accumulation(dem_proj)
    ridge      = multiscale_ridgeline_tpi(dem_proj, fine_radius_m=200.0, coarse_radius_m=1000.0)

    min_release_area_m2 = 150 * 150   # meters
    max_release_area_m2 = 4000 * 2000  # cap at 4km×2 km
    min_release_pixels = area_m2_to_pixels(dem_proj, min_release_area_m2)
    max_release_pixels = area_m2_to_pixels(dem_proj, max_release_area_m2)

    # generate start area
    release_mask = generate_release_mask(
        slope=ds['slope'],
        forest_cover=ds['fcf'],
        aspect=ds['aspect'],
        aspect_threshold=np.pi/4,   # 45° — splits across aspect sectors
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=min_release_pixels,
        max_group_size=max_release_pixels,
        smooth=True,
        reference=dem_proj,
        flow_accum=flow_accum,
        max_flow_accum=1000,
        tpi=tpi,
        tpi_split_threshold=20.0,
        strict_tpi_split_threshold=10.0,
        curvature=curv,
        curvature_split_threshold=1.0,
        strict_curvature_split_threshold=0.5,
        tpi_fine=ridge['tpi_fine'],
        tpi_coarse=ridge['tpi_coarse'],
        tpi_ridge_threshold=2.0,
        tpi_coarse_min=-5.0,
        strict_tpi_ridge_threshold=5.0,
    )

    # run FlowPy
    cell_counts_da, runout_angle_da, paths_gdf = run_flowpy_on_mask(
        dem=dem_proj,
        release_mask=release_mask,
        alpha=20,
        reference=dem_proj
    )

    # Step 3: Attach to dataset
    ds = attach_flowpy_outputs(ds, cell_counts_da, runout_angle_da, release_mask)

    return ds, paths_gdf


def attach_flowpy_outputs(ds, cell_counts, runout_angle, release_mask):
    ds['cell_counts'] = cell_counts.rio.reproject_match(ds)
    ds['cell_counts'].attrs = {'units': 'count', 'source': 'flowpy', 'product': 'cell_count'}
    ds['runout_angle'] = np.deg2rad(runout_angle.rio.reproject_match(ds))
    ds['runout_angle'].attrs = {'units': 'radians', 'source': 'flowpy', 'product': 'flow_path_alpha_angle'}
    ds['release_zones'] = release_mask.rio.reproject_match(ds)
    ds['release_zones'].attrs = {'units': 'binary', 'source': 'flowpy', 'product': 'avi_start_zones'}

    return ds


def run_flowpy_on_mask(
    dem: xr.DataArray,
    release_mask: xr.DataArray,
    alpha: float = 20,
    reference: xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Run FlowPy on a release mask and return cell counts & runout angle as DataArrays.
    """
    n_release = int((release_mask > 0).sum())
    log.info("run_flowpy_on_mask: release zone count=%d, DEM shape=%s, alpha=%s",
             n_release, dem.shape, alpha)

    if n_release == 0:
        log.info("run_flowpy_on_mask: no release zones found, returning early")

    cell_counts, runout_angle, path_list = run_flowpy(dem=dem, release=release_mask, alpha=alpha)

    # wrap outputs as DataArrays aligned to reference
    cell_counts_da = xr.zeros_like(reference)
    cell_counts_da.data = cell_counts

    runout_angle_da = xr.zeros_like(reference)
    runout_angle_da.data = runout_angle


    # concat path lists into a GeoDataFrame for export
    paths_gdf = gpd.GeoDataFrame(pd.concat(path_list, ignore_index=True), crs=path_list[0].crs)

    if reference is not None:
        cell_counts_da = cell_counts_da.rio.reproject_match(reference)
        runout_angle_da = runout_angle_da.rio.reproject_match(reference)
        paths_gdf = paths_gdf.to_crs(reference.rio.crs)

    return cell_counts_da, runout_angle_da, paths_gdf

def generate_release_mask(
    slope: xr.DataArray,
    forest_cover: xr.DataArray,
    min_slope_deg: float = 25,
    max_slope_deg: float = 60,
    max_fcf: float = 10,
    min_group_size: int = 10,
    max_group_size: int = None,
    smooth: bool = True,
    aspect: xr.DataArray = None,
    aspect_threshold: float = np.pi / 4,   # 45° — splits across aspect sectors, not just N↔S ridges
    strict_aspect_threshold: float = np.pi / 8,   # 22.5° — tighter split for oversized zones
    strict_min_slope_deg: float = 30,
    strict_max_slope_deg: float = 45,
    reference: xr.DataArray = None,
    flow_accum: xr.DataArray = None,
    max_flow_accum: float = None,
    tpi: xr.DataArray = None,
    min_tpi: float = None,
    max_tpi: float = None,
    tpi_split_threshold: float = None,
    strict_tpi_split_threshold: float = None,
    curvature: xr.DataArray = None,
    curvature_split_threshold: float = None,
    strict_curvature_split_threshold: float = None,
    tpi_fine: xr.DataArray = None,
    tpi_coarse: xr.DataArray = None,
    tpi_ridge_threshold: float = 2.0,
    tpi_coarse_min: float = -5.0,
    strict_tpi_ridge_threshold: float = None,
    strict_tpi_coarse_min: float = None,
) -> xr.DataArray:
    """
    Generate a release mask based on slope, forest cover, and optional
    ridgeline/aspect splitting.

    Parameters
    ----------
    aspect_threshold : float
        Maximum circular aspect difference (radians) allowed between adjacent
        pixels before they are split into separate release zones.  π/4 (45°)
        catches both true ridgelines and subtle terrain breaks between aspect
        sectors.  Increase toward π/2 if the mask becomes too fragmented.
    max_group_size : int or None
        If set, blobs larger than this pixel count are re-processed with
        stricter aspect and slope criteria rather than discarded outright.
    strict_aspect_threshold : float
        Aspect threshold used when re-processing oversized zones (default π/8).
    strict_min_slope_deg, strict_max_slope_deg : float
        Slope range used when re-processing oversized zones (default 30–45°).
    flow_accum : xr.DataArray or None
        D8 flow accumulation raster.  Used to exclude channel pixels.
    max_flow_accum : float or None
        Pixels with flow accumulation above this threshold are excluded.
    tpi : xr.DataArray or None
        Topographic Position Index raster.
    min_tpi : float or None
        Pixels with TPI below this value are excluded (valley floors).
    max_tpi : float or None
        Pixels with TPI above this value are excluded (ridge crests).
    tpi_split_threshold : float or None
        Absolute TPI difference between adjacent pixels that triggers a split.
    strict_tpi_split_threshold : float or None
        Tighter TPI split threshold used when re-processing oversized zones.
    curvature : xr.DataArray or None
        Total curvature raster.
    curvature_split_threshold : float or None
        Absolute curvature difference that triggers a split.
    strict_curvature_split_threshold : float or None
        Tighter curvature split threshold used when re-processing oversized zones.
    tpi_fine : xr.DataArray or None
        Fine-scale TPI (from ``multiscale_ridgeline_tpi``).  Used together
        with ``tpi_coarse`` to detect ridgeline pixels for splitting.
    tpi_coarse : xr.DataArray or None
        Coarse-scale TPI (from ``multiscale_ridgeline_tpi``).
    tpi_ridge_threshold : float
        Minimum fine-scale TPI (metres) to classify a pixel as a ridgeline
        (default 2.0 m).
    tpi_coarse_min : float
        Coarse-scale TPI must exceed this value; rejects valley-side bumps
        (default −5.0 m).
    strict_tpi_ridge_threshold : float or None
        Stricter fine-TPI threshold for oversized zone reprocessing.
        Falls back to ``tpi_ridge_threshold`` if ``None``.
    strict_tpi_coarse_min : float or None
        Stricter coarse-TPI minimum for oversized zone reprocessing.
        Falls back to ``tpi_coarse_min`` if ``None``.
    """

    # ── 1. Base slope + forest mask ──────────────────────────────────────────
    mask = (
        (slope > np.deg2rad(min_slope_deg)) &
        (slope < np.deg2rad(max_slope_deg)) &
        (forest_cover < max_fcf)
    ).astype(float)

    if smooth:
        mask = spatial_smooth(mask).round()

    if reference is not None:
        mask = mask.rio.reproject_match(reference)

    # ── 2. Reproject terrain layers to mask grid (once) ──────────────────────
    fa_proj        = flow_accum.rio.reproject_match(mask) if flow_accum is not None else None
    tpi_proj       = tpi.rio.reproject_match(mask)        if tpi       is not None else None
    curv_proj      = curvature.rio.reproject_match(mask)  if curvature is not None else None
    tpif_proj      = tpi_fine.rio.reproject_match(mask)   if tpi_fine  is not None else None
    tpic_proj      = tpi_coarse.rio.reproject_match(mask) if tpi_coarse is not None else None

    # ── 3. Flow accumulation filter ───────────────────────────────────────────
    if fa_proj is not None and max_flow_accum is not None:
        fa_vals = fa_proj.values
        exclude = np.isfinite(fa_vals) & (fa_vals > max_flow_accum)
        log.debug(
            "generate_release_mask: flow_accum filter excluded %d pixels (max_flow_accum=%g)",
            exclude.sum(), max_flow_accum,
        )
        mask_arr_fa = mask.values.copy()
        mask_arr_fa[exclude] = 0.0
        mask = xr.DataArray(mask_arr_fa, dims=mask.dims, coords=mask.coords)

    # ── 4. TPI filter ─────────────────────────────────────────────────────────
    if tpi_proj is not None:
        tpi_vals = tpi_proj.values
        mask_arr_tpi = mask.values.copy()
        if max_tpi is not None:
            mask_arr_tpi[np.isfinite(tpi_vals) & (tpi_vals > max_tpi)] = 0.0
        if min_tpi is not None:
            mask_arr_tpi[np.isfinite(tpi_vals) & (tpi_vals < min_tpi)] = 0.0
        mask = xr.DataArray(mask_arr_tpi, dims=mask.dims, coords=mask.coords)

    # ── 5. Ridgeline / aspect-break splitting ─────────────────────────────────
    # Zero out pixels on both sides of any edge where aspect changes more than
    # aspect_threshold.  This creates a clean 2-pixel-wide barrier that scipy
    # 4-connectivity cannot bridge, reliably separating adjacent slope faces.
    #
    # NOTE: no binary_closing after this step — closing would bridge the ridge
    # gaps we just created and re-merge zones across ridgelines.
    if aspect is not None:
        aspect = aspect.rio.reproject_match(mask)
        a = aspect.values

        # Circular difference between each horizontally / vertically adjacent pair
        diff_h = np.abs(np.arctan2(np.sin(a[:, :-1] - a[:, 1:]),
                                    np.cos(a[:, :-1] - a[:, 1:])))
        diff_v = np.abs(np.arctan2(np.sin(a[:-1, :] - a[1:, :]),
                                    np.cos(a[:-1, :] - a[1:, :])))

        # Mark both pixels of each high-diff edge so the barrier is at least
        # 2 pixels wide (sufficient to split with 4-connected labeling).
        ridge_h = (
            np.pad(diff_h > aspect_threshold, ((0,0),(0,1)), constant_values=False) |
            np.pad(diff_h > aspect_threshold, ((0,0),(1,0)), constant_values=False)
        )
        ridge_v = (
            np.pad(diff_v > aspect_threshold, ((0,1),(0,0)), constant_values=False) |
            np.pad(diff_v > aspect_threshold, ((1,0),(0,0)), constant_values=False)
        )
        ridge_mask = ridge_h | ridge_v

        # Dilate the ridge barrier by 1 pixel so the final gap is at least
        # 3 pixels wide (2 from bilateral marking + 1 from dilation).
        ridge_mask = binary_dilation(ridge_mask, structure=np.ones((3, 3), dtype=bool))

        log.debug("generate_release_mask: ridge pixels=%d (%.1f%% of mask)",
                  ridge_mask.sum(), 100 * ridge_mask.mean())

        mask_arr = mask.values.astype(bool)
        mask_arr[ridge_mask] = False

        # A 2×2 closing heals single noisy-aspect holes punched through valid
        # slope faces.  It can only bridge 1-pixel gaps — smaller than the
        # 3-pixel ridge barrier above — so ridgelines are never reconnected.
        mask_arr = binary_closing(mask_arr, structure=np.ones((2, 2), dtype=bool))

        mask = xr.DataArray(
            mask_arr.astype(float),
            dims=mask.dims,
            coords=mask.coords,
        )

    # ── 5.5. Multi-scale TPI ridgeline splitting ──────────────────────────────
    # Pixels classified as ridgelines (high fine-TPI, not deep in a large valley)
    # are dilated into a 3-pixel barrier and zeroed out, exactly like the aspect
    # ridge barrier above.  A 2×2 closing follows to heal single-pixel noise holes.
    if tpif_proj is not None and tpic_proj is not None:
        tpi_ridge = (
            (tpif_proj.values > tpi_ridge_threshold) &
            (tpic_proj.values > tpi_coarse_min)
        )
        tpi_ridge_barrier = binary_dilation(tpi_ridge, structure=np.ones((3, 3), dtype=bool))
        log.debug(
            "generate_release_mask: TPI ridgeline barrier=%d pixels (threshold=%.1f m, coarse_min=%.1f m)",
            tpi_ridge_barrier.sum(), tpi_ridge_threshold, tpi_coarse_min,
        )
        mask_arr = mask.values.astype(bool)
        mask_arr[tpi_ridge_barrier] = False
        mask_arr = binary_closing(mask_arr, structure=np.ones((2, 2), dtype=bool))
        mask = xr.DataArray(mask_arr.astype(float), dims=mask.dims, coords=mask.coords)

    # ── 6. Curvature splitting ────────────────────────────────────────────────
    if curv_proj is not None and curvature_split_threshold is not None:
        mask_arr = mask.values.astype(bool)
        mask_arr = _apply_linear_split(mask_arr, curv_proj.values, curvature_split_threshold)
        mask = xr.DataArray(mask_arr.astype(float), dims=mask.dims, coords=mask.coords)

    # ── 7. TPI splitting ──────────────────────────────────────────────────────
    if tpi_proj is not None and tpi_split_threshold is not None:
        mask_arr = mask.values.astype(bool)
        mask_arr = _apply_linear_split(mask_arr, tpi_proj.values, tpi_split_threshold)
        mask = xr.DataArray(mask_arr.astype(float), dims=mask.dims, coords=mask.coords)

    # ── 8. Size filtering — keep normal zones, re-process oversized ones ─────
    # Apply DEM nodata mask BEFORE labeling so NaN holes don't fragment zones
    # after the size filter has already run (which would let sub-threshold
    # shards through into flowpy).
    arr = mask.values.astype(bool)
    if reference is not None:
        arr &= np.isfinite(reference.values)
    labeled, _ = ndlabel(arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background

    # Normal-sized zones pass the standard min/max filter
    keep_normal = sizes >= min_group_size
    if max_group_size is not None:
        keep_normal &= sizes <= max_group_size
    normal_arr = np.isin(labeled, np.where(keep_normal)[0])

    # Oversized zones: re-run with stricter aspect + slope instead of discarding
    strict_arr = np.zeros_like(normal_arr)
    if max_group_size is not None and aspect is not None:
        large_labels = np.where((sizes > 0) & (sizes > max_group_size))[0]
        if len(large_labels) > 0:
            log.debug(
                "generate_release_mask: %d oversized zones — re-processing with "
                "strict_aspect=%.4f rad, slope=[%g, %g]°",
                len(large_labels), strict_aspect_threshold,
                strict_min_slope_deg, strict_max_slope_deg,
            )
            large_region = np.isin(labeled, large_labels)

            # Strict slope filter (reproject slope to match mask CRS)
            slope_proj = slope.rio.reproject_match(mask)
            strict_slope = (
                (slope_proj.values > np.deg2rad(strict_min_slope_deg)) &
                (slope_proj.values < np.deg2rad(strict_max_slope_deg))
            )

            # Strict aspect-based ridgeline splitting — reuse `a` from step 2
            diff_h = np.abs(np.arctan2(np.sin(a[:, :-1] - a[:, 1:]),
                                        np.cos(a[:, :-1] - a[:, 1:])))
            diff_v = np.abs(np.arctan2(np.sin(a[:-1, :] - a[1:, :]),
                                        np.cos(a[:-1, :] - a[1:, :])))
            strict_ridge_h = (
                np.pad(diff_h > strict_aspect_threshold, ((0,0),(0,1)), constant_values=False) |
                np.pad(diff_h > strict_aspect_threshold, ((0,0),(1,0)), constant_values=False)
            )
            strict_ridge_v = (
                np.pad(diff_v > strict_aspect_threshold, ((0,1),(0,0)), constant_values=False) |
                np.pad(diff_v > strict_aspect_threshold, ((1,0),(0,0)), constant_values=False)
            )
            strict_ridge = binary_dilation(
                strict_ridge_h | strict_ridge_v, structure=np.ones((3, 3), dtype=bool)
            )

            reprocessed = large_region & strict_slope & ~strict_ridge

            if tpif_proj is not None and tpic_proj is not None:
                rt = strict_tpi_ridge_threshold if strict_tpi_ridge_threshold is not None else tpi_ridge_threshold
                cm = strict_tpi_coarse_min      if strict_tpi_coarse_min      is not None else tpi_coarse_min
                tpi_ridge = (
                    (tpif_proj.values > rt) &
                    (tpic_proj.values > cm)
                )
                tpi_ridge_dilated = binary_dilation(tpi_ridge, structure=np.ones((3, 3), dtype=bool))
                reprocessed = reprocessed & ~tpi_ridge_dilated
                reprocessed = binary_closing(reprocessed, structure=np.ones((2, 2), dtype=bool))

            if curv_proj is not None and (curvature_split_threshold or strict_curvature_split_threshold):
                thresh = strict_curvature_split_threshold or curvature_split_threshold
                reprocessed = _apply_linear_split(reprocessed, curv_proj.values, thresh)

            if tpi_proj is not None and (tpi_split_threshold or strict_tpi_split_threshold):
                thresh = strict_tpi_split_threshold or tpi_split_threshold
                reprocessed = _apply_linear_split(reprocessed, tpi_proj.values, thresh)

            strict_da = xr.DataArray(reprocessed.astype(float), dims=mask.dims, coords=mask.coords)
            strict_arr = filter_pixel_groups(strict_da, min_size=min_group_size).values.astype(bool)
            log.debug(
                "generate_release_mask: strict re-processing retained %d pixels from oversized zones",
                strict_arr.sum(),
            )

    combined = (normal_arr | strict_arr).astype(float)
    da_out = xr.zeros_like(reference)
    da_out.data = combined

    return da_out