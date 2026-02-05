import numpy as np
import xarray as xr

from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

def generate_runcount_alpha_angle(ds):
    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())

    min_release_area_m2 = 300 *300 # meteers
    min_release_pixels = area_m2_to_pixels(dem_proj, min_release_area_m2)

    # generate start area
    release_mask = generate_release_mask(
        slope=ds['slope'],
        forest_cover=ds['fcf'],
        min_slope_deg=35,
        max_slope_deg=45,
        max_fcf=10,
        min_group_size=min_release_pixels,
        smooth=True,
        reference=dem_proj
    )

    # run FlowPy
    cell_counts_da, runout_angle_da = run_flowpy_on_mask(
        dem=dem_proj,
        release_mask=release_mask,
        alpha=25,
        reference=dem_proj
    )

    # Step 3: Attach to dataset
    ds = attach_flowpy_outputs(ds, cell_counts_da, runout_angle_da)

    return ds


def attach_flowpy_outputs(ds, cell_counts, runout_angle):
    ds['cell_counts'] = cell_counts.rio.reproject_match(ds)
    ds['cell_counts'].attrs = {'units': 'count', 'source': 'flowpy', 'product': 'cell_count'}
    ds['runout_angle'] = np.deg2rad(runout_angle.rio.reproject_match(ds))
    ds['runout_angle'].attrs = {'units': 'radians', 'source': 'flowpy', 'product': 'flow_path_alpha_angle'}

    return ds


def generate_release_mask(
    slope: xr.DataArray,
    forest_cover: xr.DataArray,
    min_slope_deg: float = 35,
    max_slope_deg: float = 45,
    max_fcf: float = 10,
    min_group_size: int = 300,
    smooth: bool = True,
    reference: xr.DataArray = None,
) -> xr.DataArray:
    """
    Generate a release mask based on slope, forest cover, and optional filtering.

    Returns a DataArray of 0/1 values aligned to `reference` CRS/grid.
    """
    mask = (
        (slope > np.deg2rad(min_slope_deg)) &
        (slope < np.deg2rad(max_slope_deg)) &
        (forest_cover < max_fcf)
    ).astype(float)

    if smooth:
        mask = spatial_smooth(mask).round()


    if reference is not None:
        mask = mask.rio.reproject_match(reference)

    da_out = xr.zeros_like(reference)
    da_out.data = filter_pixel_groups(mask, min_size=min_group_size)
    return da_out


def run_flowpy_on_mask(
    dem: xr.DataArray,
    release_mask: xr.DataArray,
    alpha: float = 25,
    reference: xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Run FlowPy on a release mask and return cell counts & runout angle as DataArrays.
    """
    cell_counts, runout_angle = run_flowpy(dem=dem, release=release_mask, alpha=alpha)

    # wrap outputs as DataArrays aligned to reference
    cell_counts_da = xr.zeros_like(reference)
    cell_counts_da.data = cell_counts

    runout_angle_da = xr.zeros_like(reference)
    runout_angle_da.data = runout_angle

    if reference is not None:
        cell_counts_da = cell_counts_da.rio.reproject_match(reference)
        runout_angle_da = runout_angle_da.rio.reproject_match(reference)

    return cell_counts_da, runout_angle_da
