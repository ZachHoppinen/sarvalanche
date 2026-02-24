import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_closing

from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

def generate_runcount_alpha_angle(ds):
    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())

    min_release_area_m2 = 50 * 50 # meters
    min_release_pixels = area_m2_to_pixels(dem_proj, min_release_area_m2)

    # generate start area
    release_mask = generate_release_mask(
        slope=ds['slope'],
        forest_cover=ds['fcf'],
        aspect = ds['aspect'],
        aspect_threshold=np.pi/2,
        min_slope_deg=25,
        max_slope_deg=60,
        max_fcf=10,
        min_group_size=min_release_pixels,
        smooth=True,
        reference=dem_proj
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

# def generate_release_mask(
#     slope: xr.DataArray,
#     forest_cover: xr.DataArray,
#     min_slope_deg: float = 25,
#     max_slope_deg: float = 60,
#     max_fcf: float = 10,
#     min_group_size: int = 100,
#     smooth: bool = True,
#     aspect: xr.DataArray = None,
#     aspect_threshold = np.pi/8,
#     reference: xr.DataArray = None,
# ) -> xr.DataArray:
#     """
#     Generate a release mask based on slope, forest cover, and optional filtering.

#     Returns a DataArray of 0/1 values aligned to `reference` CRS/grid.
#     """
#     mask = (
#         (slope > np.deg2rad(min_slope_deg)) &
#         (slope < np.deg2rad(max_slope_deg)) &
#         (forest_cover < max_fcf)
#     ).astype(float)

#     if smooth:
#         mask = spatial_smooth(mask).round()

#     if reference is not None:
#         mask = mask.rio.reproject_match(reference)

#     if aspect is not None:
#         aspect = aspect.rio.reproject_match(mask)
#         a = aspect.values  # (2585, 1674) pure numpy
#         diff_h = np.abs(np.arctan2(np.sin(a[:, :-1] - a[:, 1:]),
#                                     np.cos(a[:, :-1] - a[:, 1:])))
#         diff_v = np.abs(np.arctan2(np.sin(a[:-1, :] - a[1:, :]),
#                                     np.cos(a[:-1, :] - a[1:, :])))

#         aspect_break = (
#             np.pad(diff_h > aspect_threshold, ((0,0),(0,1)), constant_values=False) |
#             np.pad(diff_h > aspect_threshold, ((0,0),(1,0)), constant_values=False) |
#             np.pad(diff_v > aspect_threshold, ((0,1),(0,0)), constant_values=False) |
#             np.pad(diff_v > aspect_threshold, ((1,0),(0,0)), constant_values=False)
#         )
#         mask = mask.astype(bool) & ~xr.DataArray(aspect_break, dims=mask.dims, coords=mask.coords)
#         mask = mask.astype(float)

#     da_out = xr.zeros_like(reference)
#     da_out.data = filter_pixel_groups(mask, min_size=min_group_size)

#     valid_mask = np.isfinite(reference.values)
#     # da_out = da_out.where(valid_mask)
#     da_out = da_out.where(valid_mask, other=0.0)

#     return da_out

def run_flowpy_on_mask(
    dem: xr.DataArray,
    release_mask: xr.DataArray,
    alpha: float = 25,
    reference: xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Run FlowPy on a release mask and return cell counts & runout angle as DataArrays.
    """
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
    smooth: bool = True,
    aspect: xr.DataArray = None,
    aspect_threshold: float = np.pi / 2,   # ~90° — only split at true ridgelines
    reference: xr.DataArray = None,
) -> xr.DataArray:
    """
    Generate a release mask based on slope, forest cover, and optional
    ridgeline splitting.

    Changes from previous version:
    - aspect_threshold default raised to π/2 (90°) — only breaks at real
      ridgelines, not noisy slope interior variation
    - binary_closing applied after aspect splitting to reconnect fragments
      caused by single noisy pixels
    - aspect break used as a hard barrier (sets mask=0 at break pixels)
      rather than fragmenting via 4-directional padding, which was too aggressive
    - filter_pixel_groups runs last, after closing, so min_group_size acts
      on the final connected regions not pre-close fragments
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

    # ── 2. Ridgeline splitting via aspect breaks ─────────────────────────────
    # Only break at pixels where aspect changes sharply (true ridgelines),
    # not at the minor variation within a uniform slope face.
    # At 30m resolution, π/2 (90°) is a reasonable threshold — a ridge
    # separating N-facing and S-facing slopes will exceed this easily,
    # but normal within-slope aspect noise (~10-30°) will not.
    if aspect is not None:
        aspect = aspect.rio.reproject_match(mask)
        a = aspect.values

        # Compute angular difference between adjacent pixels
        diff_h = np.abs(np.arctan2(np.sin(a[:, :-1] - a[:, 1:]),
                                    np.cos(a[:, :-1] - a[:, 1:])))
        diff_v = np.abs(np.arctan2(np.sin(a[:-1, :] - a[1:, :]),
                                    np.cos(a[:-1, :] - a[1:, :])))

        # A pixel is a ridgeline if it borders a sharp aspect change
        # Use OR of both sides so the ridge pixel itself gets masked
        ridge_h = (
            np.pad(diff_h > aspect_threshold, ((0,0),(0,1)), constant_values=False) |
            np.pad(diff_h > aspect_threshold, ((0,0),(1,0)), constant_values=False)
        )
        ridge_v = (
            np.pad(diff_v > aspect_threshold, ((0,1),(0,0)), constant_values=False) |
            np.pad(diff_v > aspect_threshold, ((1,0),(0,0)), constant_values=False)
        )
        ridge_mask = ridge_h | ridge_v

        # Zero out ridge pixels in the release mask — this splits zones
        # at ridgelines without fragmenting interior slope pixels
        mask_arr = mask.values.astype(bool)
        mask_arr[ridge_mask] = False

        # ── 3. Close small gaps caused by noisy ridge pixels ─────────────────
        # A single noisy aspect pixel can punch a hole through a valid slope.
        # binary_closing with a 3x3 structure reconnects fragments separated
        # by 1-2 pixel gaps while preserving true ridgeline breaks (which are
        # wider than 1-2 pixels at 30m)
        structure = np.ones((5, 5), dtype=bool)
        mask_arr = binary_closing(mask_arr, structure=structure)

        mask = xr.DataArray(
            mask_arr.astype(float),
            dims=mask.dims,
            coords=mask.coords,
        )

    # ── 4. Remove small disconnected patches ─────────────────────────────────
    da_out = xr.zeros_like(reference)
    da_out.data = filter_pixel_groups(mask, min_size=min_group_size)

    valid_mask = np.isfinite(reference.values)
    da_out = da_out.where(valid_mask, other=0.0)

    return da_out