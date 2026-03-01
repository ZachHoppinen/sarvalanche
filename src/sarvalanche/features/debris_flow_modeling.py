import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.ndimage import binary_dilation, binary_fill_holes, label as ndlabel

from sarvalanche.features.ridgelines import generate_ridgelines
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

log = logging.getLogger(__name__)


def generate_runcount_alpha_angle(ds):
    from sarvalanche.utils.terrain import compute_flow_accumulation

    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())
    log.debug("generate_runcount_alpha_angle: DEM shape=%s", dem_proj.shape)

    flow_accum = compute_flow_accumulation(dem_proj)

    # generate start area
    release_mask = generate_release_mask_simple(
        slope=ds['slope'],
        dem=dem_proj,
        flow_accum=flow_accum,
        forest_cover=ds['fcf'],
        max_flow_accum_channel=10,
        max_fcf=20,
        reference=dem_proj,
    )

    # run FlowPy
    cell_counts_da, runout_angle_da, paths_gdf = run_flowpy_on_mask(
        dem=dem_proj,
        release_mask=release_mask,
        alpha=20,
        reference=dem_proj
    )

    # Attach to dataset
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
) -> tuple[xr.DataArray, xr.DataArray, gpd.GeoDataFrame]:
    """Run FlowPy on a release mask and return cell counts, runout angle, and paths."""
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


def generate_release_mask_simple(
    slope: xr.DataArray,
    dem: xr.DataArray,
    flow_accum: xr.DataArray,
    forest_cover: xr.DataArray = None,
    min_slope_deg: float = 32.0,
    max_slope_deg: float = 60.0,
    max_fcf: float = 5.0,
    max_flow_accum_channel: float = 10.0,
    smooth: bool = True,
    tpi_radius_m: float = 300.0,
    tpi_threshold: float = 5.0,
    curv_threshold: float = -1.0,
    ridge_smooth_sigma: float = 2.0,
    ridge_barrier_width: int = 3,
    min_release_area_m2: float = 100 * 100,
    max_release_area_m2: float = 5000 * 5000,
    reference: xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Generate release zone mask using slope, forest cover, flow accumulation,
    and Hessian-based ridgeline splitting.

    Parameters
    ----------
    slope : xr.DataArray
        Slope raster in radians. Must have a projected CRS.
    dem : xr.DataArray
        Elevation raster (projected CRS) for ridgeline detection.
    flow_accum : xr.DataArray
        D8 flow accumulation (number of upstream cells).
    forest_cover : xr.DataArray, optional
        Forest cover fraction (0-100 scale).
    min_slope_deg, max_slope_deg : float
        Slope range in degrees for the base mask.
    max_fcf : float
        Pixels with forest cover >= max_fcf are excluded.
    max_flow_accum_channel : float
        Pixels with flow accumulation above this are excluded.
    smooth : bool
        Apply spatial smoothing to the base mask before ridgeline splitting.
    tpi_radius_m : float
        TPI neighbourhood radius for ridgeline detection (metres).
    tpi_threshold : float
        Minimum TPI to qualify as high ground for ridgeline detection.
    curv_threshold : float
        Maximum lambda2 curvature for ridge classification.
    ridge_smooth_sigma : float
        Gaussian smoothing sigma for Hessian computation (pixels).
    ridge_barrier_width : int
        Dilation iterations to widen the ridgeline barrier.
    min_release_area_m2, max_release_area_m2 : float
        Size filter bounds in m².
    reference : xr.DataArray, optional
        If provided, all inputs are reprojected to this grid.

    Returns
    -------
    (release_mask, ridgelines) : tuple[xr.DataArray, xr.DataArray]
        Binary release mask and ridgeline skeleton, both aligned to *reference*.
    """
    # ── 1. Reproject ──────────────────────────────────────────────────────────
    if reference is not None:
        slope_r = slope.rio.reproject_match(reference)
        fa_r    = flow_accum.rio.reproject_match(reference)
        dem_r   = dem.rio.reproject_match(reference)
        fcf_r   = forest_cover.rio.reproject_match(reference) if forest_cover is not None else None
    else:
        slope_r = slope
        fa_r    = flow_accum
        dem_r   = dem
        fcf_r   = forest_cover

    fa_vals = fa_r.values

    # ── 2. Base slope mask ────────────────────────────────────────────────────
    mask_arr = (
        (slope_r.values > np.deg2rad(min_slope_deg)) &
        (slope_r.values < np.deg2rad(max_slope_deg))
    )
    log.debug("generate_release_mask_simple: slope filter [%.0f, %.0f]° -> %d px",
              min_slope_deg, max_slope_deg, mask_arr.sum())

    # ── 3. Forest cover exclusion ─────────────────────────────────────────────
    if fcf_r is not None:
        mask_arr &= fcf_r.values < max_fcf
        log.debug("generate_release_mask_simple: after FCF filter (max_fcf=%.0f): %d px",
                  max_fcf, mask_arr.sum())

    # ── 4. Channel exclusion ──────────────────────────────────────────────────
    mask_arr &= ~(np.isfinite(fa_vals) & (fa_vals > max_flow_accum_channel))
    log.debug("generate_release_mask_simple: after channel filter: %d px", mask_arr.sum())

    # ── 5. Spatial smoothing ──────────────────────────────────────────────────
    if smooth:
        mask_da  = xr.DataArray(mask_arr.astype(float), dims=slope_r.dims, coords=slope_r.coords)
        mask_arr = spatial_smooth(mask_da).round().values.astype(bool)
        log.debug("generate_release_mask_simple: after smoothing: %d px", mask_arr.sum())

    # ── 6. Ridgeline detection (pixel size inferred from DEM) ─────────────────
    ridgelines = generate_ridgelines(
        dem_r,
        tpi_radius_m=tpi_radius_m,
        smooth_sigma=ridge_smooth_sigma,
        tpi_threshold=tpi_threshold,
        curv_threshold=curv_threshold,
    )

    # ── 7. Dilate skeleton into barrier and cut ───────────────────────────────
    barrier = binary_dilation(
        ridgelines.values.astype(bool),
        iterations=ridge_barrier_width,
    )
    mask_arr[barrier] = False
    log.debug("generate_release_mask_simple: after ridgeline barrier (width=%d): %d px",
              ridge_barrier_width, mask_arr.sum())

    # ── 8. Remove isolated single pixels ─────────────────────────────────────
    labeled, _ = ndlabel(mask_arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    single_labels = np.where(sizes == 1)[0]
    mask_arr[np.isin(labeled, single_labels)] = False
    log.debug("generate_release_mask_simple: removed %d isolated pixels", len(single_labels))

    # ── 9. Fill enclosed holes ────────────────────────────────────────────────
    before = int(mask_arr.sum())
    mask_arr = binary_fill_holes(mask_arr)
    log.debug("generate_release_mask_simple: hole filling added %d px",
              int(mask_arr.sum()) - before)

    # ── 10. Size filter ───────────────────────────────────────────────────────
    min_px = area_m2_to_pixels(dem_r, min_release_area_m2)
    max_px = area_m2_to_pixels(dem_r, max_release_area_m2)
    labeled, _ = ndlabel(mask_arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = (sizes >= min_px) & (sizes <= max_px)
    mask_arr = keep[labeled]
    log.debug("generate_release_mask_simple: size filter [%d, %d] px -> %d zones, %d px",
              min_px, max_px, int(keep.sum()), int(mask_arr.sum()))

    # ── 11. Wrap and return ───────────────────────────────────────────────────
    if reference is not None:
        out = xr.zeros_like(reference)
        out.data = mask_arr.astype(float)
    else:
        out = xr.DataArray(
            mask_arr.astype(float), dims=slope_r.dims, coords=slope_r.coords
        )
    return out