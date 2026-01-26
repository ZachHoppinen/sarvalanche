from .BaseDataSource import BaseDataSource

import os
from pathlib import Path
import xarray as xr
import rioxarray

from sarvalanche.utils.validation import validate_canonical

class DemSource(BaseDataSource):
    sensor = "DEM"
    product = "Elevation"

    def __init__(self, *, cache_dir=None):
        self.cache_dir = cache_dir


    def load(
        self,
        *,
        aoi,
        dem_path: str | Path | None = None,
        resolution: float = 10,
        epsg: int = 4326,
        ref_da: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """
        Load a DEM either from a user-supplied file or from py3dep.

        Parameters
        ----------
        aoi : shapely geometry or GeoDataFrame
        dem_path : optional path to a DEM GeoTIFF
        resolution : output resolution in meters
        epsg : target CRS EPSG code
        """

        if dem_path is not None:
            dem_path = Path(dem_path)

            if not dem_path.exists():
                raise FileNotFoundError(dem_path)

            dem = xr.open_dataarray(dem_path).squeeze(drop=True)

            # Reproject / resample to requested grid
            # dem = dem.rio.reproject(
            #     f"EPSG:{epsg}",
            #     resolution=resolution,
            # )

        else:
            os.environ["HYRIVER_CACHE_NAME"] = self.cache_dir
            import py3dep
            dem = py3dep.get_map(
                "DEM",
                geometry=aoi,
                resolution=resolution,
                crs=f"EPSG:{epsg}",
            )

        if ref_da is not None:
            dem = dem.rio.reproject_match(ref_da)

        dem.name = "dem"
        dem.attrs.update(
            sensor = self.sensor,
            product = self.product,
            source="user" if dem_path else "py3dep",
            resolution=resolution,
            crs = epsg,
            units = 'meters'
        )

        validate_canonical(dem)

        return dem
