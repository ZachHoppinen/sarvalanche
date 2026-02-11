"""
Exclusion mask application for the avalanche detection pipeline.

This function applies water and urban masks to exclude non-avalanche terrain.
Should be called AFTER pixelwise probabilities but BEFORE dense CRF grouping.
"""

import logging

log = logging.getLogger(__name__)


def apply_exclusion_masks(ds, apply_water=True, apply_urban=True, year=2021):
    """
    Apply water and urban masks to exclude non-avalanche terrain.

    This function should be called AFTER pixelwise probabilities are calculated
    but BEFORE the dense CRF grouping step. This ensures that the spatial
    smoothing algorithm doesn't spread detections into invalid terrain like
    lakes or cities.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with p_pixelwise already calculated. Must have:
        - p_pixelwise: pixel-by-pixel probabilities
        - Proper CRS and bounds set via rioxarray
    apply_water : bool, optional
        If True, exclude water bodies. Default is True.
    apply_urban : bool, optional
        If True, exclude developed/urban areas. Default is True.
    year : int, optional
        NLCD year to use for land cover data. Default is 2021.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - p_pixelwise: masked to exclude invalid terrain (set to 0)
        - mask_water: water mask (if apply_water=True)
        - mask_urban: urban mask (if apply_urban=True)
        - mask_exclusion: combined exclusion mask

    Notes
    -----
    The masks are stored in the dataset for transparency and debugging.
    Excluded pixels have their p_pixelwise set to 0, which effectively
    removes them from further processing.

    Exclusion criteria:
    - Water: NLCD class 11 (Open Water) - lakes, rivers, reservoirs
    - Urban: NLCD classes 21-24 (all development levels)

    Examples
    --------
    >>> # In your pipeline, between steps 6 and 7:
    >>> ds['p_pixelwise'] = get_pixelwise_probabilities(ds, avalanche_date)
    >>> ds = apply_exclusion_masks(ds)  # <- Apply masks here
    >>> ds['detections'] = group_classes(ds['p_pixelwise'], cache_dir)
    """

    if not apply_water and not apply_urban:
        log.info("No masks requested - skipping exclusion masks")
        return ds

    log.info("Applying exclusion masks to p_pixelwise")

    # Get dataset info for mask extraction
    # Note: AOI extraction depends on your dataset structure
    # You may need to adjust this based on how bounds are stored
    crs = ds.rio.crs
    ref_grid = ds['p_pixelwise']  # Use pixelwise as reference for alignment

    # Get bounds from dataset
    # Different options depending on your dataset structure:
    try:
        # Option 1: Get bounds from rioxarray
        bounds = ds.rio.bounds()
        from shapely.geometry import box
        aoi = box(*bounds)
    except:
        # Option 2: Get bounds from coordinates
        x_min, x_max = float(ds.x.min()), float(ds.x.max())
        y_min, y_max = float(ds.y.min()), float(ds.y.max())
        from shapely.geometry import box
        aoi = box(x_min, y_min, x_max, y_max)

    # Initialize exclusion mask as all zeros (all pixels valid)
    import xarray as xr
    import numpy as np
    exclusion_mask = xr.zeros_like(ds['p_pixelwise'], dtype=bool)

    # Apply water mask if requested
    if apply_water:
        try:
            log.info("Masking water extent...")
            assert 'water_mask' in ds.data_vars, f'Water masking requested but no water_mask layer.'
            exclusion_mask = exclusion_mask | (ds["water_mask"] == 1)
            n_water = (ds["water_mask"] == 1).sum().values
            log.info(f"  Water pixels: {n_water} ({n_water/ds["water_mask"].size*100:.1f}%)")
        except Exception as e:
            log.warning(f"Could not get water extent: {e}")
            log.warning("Continuing without water masking")

    # Apply urban mask if requested
    if apply_urban:
        try:
            log.info("Masking urban extent...")
            assert 'urban_mask' in ds.data_vars, f'Urban masking requested but no urban_mask layer.'
            exclusion_mask = exclusion_mask | (ds["urban_mask"] == 1)
            n_urban = (ds["urban_mask"] == 1).sum().values
            log.info(f"  Urban pixels: {n_urban} ({n_urban/ds["urban_mask"].size*100:.1f}%)")
        except Exception as e:
            log.warning(f"Could not get urban extent: {e}")
            log.warning("Continuing without urban masking")

    # Store combined exclusion mask
    ds['mask_exclusion'] = exclusion_mask.astype(int)
    ds['mask_exclusion'].attrs = {
        'description': 'Exclusion mask (1=excluded, 0=valid terrain)',
        'water_excluded': str(apply_water),
        'urban_excluded': str(apply_urban),
        'nlcd_year': str(year),
        'water_classes': '11 (Open Water)',
        'urban_classes': '21-24 (Developed)',
        'product': 'masked_pixels',
        'source': 'nlcd',
        'units': 'binary'
    }

    # Apply mask to pixelwise probabilities
    # Set excluded pixels to 0
    n_excluded = exclusion_mask.sum().values
    percent_excluded = n_excluded / exclusion_mask.size * 100

    log.info(f"Total excluded: {n_excluded} pixels ({percent_excluded:.1f}% of area)")

    # Apply the mask
    ds['p_pixelwise'] = ds['p_pixelwise'].where(~exclusion_mask, 0)

    log.info("Exclusion masks applied successfully")

    return ds
