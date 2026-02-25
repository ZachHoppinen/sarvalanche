import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation, binary_closing

from sarvalanche.masks.size_filter import filter_pixel_groups
from sarvalanche.preprocessing.spatial import spatial_smooth
from sarvalanche.utils.projections import area_m2_to_pixels
from sarvalanche.vendored.flowpy import run_flowpy

log = logging.getLogger(__name__)

def generate_runcount_alpha_angle(ds):
    # flowpy needs to run in projected coordinate system
    dem_proj = ds['dem'].rio.reproject(ds['dem'].rio.estimate_utm_crs())
    log.debug("generate_runcount_alpha_angle: DEM shape=%s", dem_proj.shape)

    min_release_area_m2 = 50 * 50   # meters
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
    reference: xr.DataArray = None,
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
        If set, blobs larger than this pixel count are discarded.  Useful to
        cap oversized zones that result from undetected ridgeline merges.
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

    # ── 2. Ridgeline / aspect-break splitting ────────────────────────────────
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

    # ── 3. Remove small / large disconnected patches ─────────────────────────
    da_out = xr.zeros_like(reference)
    da_out.data = filter_pixel_groups(mask, min_size=min_group_size, max_size=max_group_size)

    valid_mask = np.isfinite(reference.values)
    da_out = da_out.where(valid_mask, other=0.0)

    return da_out