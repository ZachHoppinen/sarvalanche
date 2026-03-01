import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.ndimage import binary_dilation, binary_fill_holes, label as ndlabel

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
        tpi_fine_threshold=20,
        tpi_fine_radius_m=150,
        tpi_coarse_threshold=55,
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

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.ndimage import binary_dilation, binary_fill_holes, label as ndlabel

from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

def generate_release_mask_simple(
    slope: xr.DataArray,
    dem: xr.DataArray,
    flow_accum: xr.DataArray,
    forest_cover: xr.DataArray = None,
    min_slope_deg: float = 28.0,
    max_slope_deg: float = 70.0,
    max_fcf: float = 5.0,
    max_flow_accum_channel: float = 10.0,
    smooth: bool = True,
    tpi_radius_m: float = 300.0,
    tpi_threshold: float = 5.0,
    curv_threshold: float = -0.5,
    ridge_smooth_sigma: float = 2.0,
    ridge_barrier_width: int = 1,       # dilation iterations for barrier
    min_release_area_m2: float = 100 * 100,
    max_release_area_m2: float = 5000 * 5000,
    pixel_size_m: float = 10.0,
    reference: xr.DataArray = None,
) -> xr.DataArray:

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

    # ── 3. Forest cover exclusion ─────────────────────────────────────────────
    if fcf_r is not None:
        mask_arr &= fcf_r.values < max_fcf

    # ── 4. Channel exclusion ──────────────────────────────────────────────────
    mask_arr &= ~(np.isfinite(fa_vals) & (fa_vals > max_flow_accum_channel))

    # ── 5. Spatial smoothing ──────────────────────────────────────────────────
    if smooth:
        mask_da  = xr.DataArray(mask_arr.astype(float), dims=slope_r.dims, coords=slope_r.coords)
        mask_arr = spatial_smooth(mask_da).round().values.astype(bool)

    # ── 6. Ridgeline detection ────────────────────────────────────────────────
    print("dem_r shape:", dem_r.shape)
    print("dem_r hash:", array_hash(dem_r.values))
    print("dem_r coords x0:", float(dem_r.x[0]), float(dem_r.x[-1]))
    print("dem_r dtype:", dem_r.dtype)

    # Inside generate_release_mask_simple, just before the generate_ridgelines call:
    print("FROM FUNCTION:")
    print(f"  tpi_radius_m={tpi_radius_m}")
    print(f"  pixel_size_m={pixel_size_m}")
    print(f"  smooth_sigma={ridge_smooth_sigma}")
    print(f"  tpi_threshold={tpi_threshold}")
    print(f"  curv_threshold={curv_threshold}")

    ridgelines = generate_ridgelines(
        dem_r,
        tpi_radius_m=tpi_radius_m,
        pixel_size_m=pixel_size_m,
        smooth_sigma=ridge_smooth_sigma,
        tpi_threshold=tpi_threshold,
        curv_threshold=curv_threshold,
    )

    # ── 7. Dilate skeleton into barrier and cut ───────────────────────────────
    # The skeleton is 1px wide — dilate to make a barrier wide enough to
    # reliably split adjacent release zones. ridge_barrier_width=3 gives
    # a 7px wide cut (3 each side + the skeleton pixel itself).
    barrier = binary_dilation(
        ridgelines.values.astype(bool),
        iterations=ridge_barrier_width
    )
    mask_arr[barrier] = False

    # ── 8. Remove isolated single pixels ─────────────────────────────────────
    labeled, _ = ndlabel(mask_arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask_arr[np.isin(labeled, np.where(sizes == 1)[0])] = False

    # ── 9. Fill enclosed holes ────────────────────────────────────────────────
    mask_arr = binary_fill_holes(mask_arr)

    # ── 10. Size filter ───────────────────────────────────────────────────────
    min_px = area_m2_to_pixels(dem_r, min_release_area_m2)
    max_px = area_m2_to_pixels(dem_r, max_release_area_m2)
    labeled, _ = ndlabel(mask_arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = (sizes >= min_px) & (sizes <= max_px)
    mask_arr = keep[labeled]

    # ── 11. Wrap and return ───────────────────────────────────────────────────
    if reference is not None:
        out = xr.zeros_like(reference)
        out.data = mask_arr.astype(float)
    else:
        out = xr.DataArray(
            mask_arr.astype(float), dims=slope_r.dims, coords=slope_r.coords
        )
    return out, ridgelines