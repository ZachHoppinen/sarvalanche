"""Replace water_mask in CNFAIC netCDFs with ESA WorldCover water layer."""

import logging
import xarray as xr
import numpy as np
from pathlib import Path
from shapely.geometry import box

from sarvalanche.io.load_data import _get_esa_water

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NC_DIR = Path("local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood")

def replace_water_mask(nc_path):
    log.info("Processing %s", nc_path.name)
    ds = xr.open_dataset(nc_path).load()  # load into memory so we can close + overwrite

    # Get AOI from dataset bounds
    bounds = ds.rio.bounds()
    aoi = box(*bounds)
    crs = ds.rio.crs

    # Use a 2D variable as ref_grid for reprojection
    if "dem" in ds:
        ref_grid = ds["dem"]
    elif "water_mask" in ds and ds["water_mask"].ndim == 2:
        ref_grid = ds["water_mask"]
    else:
        ref_grid = ds["water_mask"].isel(time=0, drop=True) if "time" in ds["water_mask"].dims else ds["water_mask"]

    # Ensure ref_grid has CRS
    if ref_grid.rio.crs is None:
        ref_grid = ref_grid.rio.write_crs(crs)

    log.info("  Old water_mask: source=%s, unique=%s, sum=%d",
             ds["water_mask"].attrs.get("source", "?"),
             np.unique(ds["water_mask"].values[~np.isnan(ds["water_mask"].values)])[:5],
             int((ds["water_mask"] == 1).sum().values))

    # Fetch ESA WorldCover water
    esa_water = _get_esa_water(aoi, str(crs), ref_grid=ref_grid)
    log.info("  ESA water pixels: %d / %d (%.1f%%)",
             int((esa_water == 1).sum().values), esa_water.size,
             (esa_water == 1).sum().values / esa_water.size * 100)

    # If original had a time dim, broadcast
    old_wm = ds["water_mask"]
    if "time" in old_wm.dims:
        esa_water = esa_water.expand_dims(time=ds.time)

    # Drop spatial_ref if ESA data brought its own (conflicts with dataset's)
    if "spatial_ref" in esa_water.coords:
        esa_water = esa_water.drop_vars("spatial_ref")

    # Replace
    ds = ds.drop_vars("water_mask")
    ds["water_mask"] = esa_water.astype(np.float64)
    ds["water_mask"].attrs = {
        "units": "binary",
        "source": "esa_worldcover",
        "product": "water_extent",
        "description": "1=water, 0=land",
    }

    # Write to temp file then rename (avoids writing to open file)
    tmp_path = nc_path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp_path)
    tmp_path.rename(nc_path)
    log.info("  Written to %s", nc_path)


def main():
    nc_files = sorted(NC_DIR.glob("season_*.nc"))
    log.info("Found %d netCDF files", len(nc_files))
    for nc in nc_files:
        replace_water_mask(nc)
    log.info("Done")


if __name__ == "__main__":
    main()
