from .BaseDataSource import BaseDataSource

from pathlib import Path
import xarray as xr
import rioxarray
import py3dep

class DEMSource(BaseDataSource):
    sensor = "DEM"
    product = "Elevation"

    def load(
        self,
        *,
        aoi,
        dem_path: str | Path | None = None,
        resolution: float = 10,
        epsg: int = 4326,
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
            dem = dem.rio.reproject(
                f"EPSG:{epsg}",
                resolution=resolution,
            )

        else:
            dem = py3dep.get_map(
                "DEM",
                geometry=aoi,
                resolution=resolution,
                crs=f"EPSG:{epsg}",
            )

        dem.name = "dem"
        dem.attrs.update(
            sensor = self.sensor
            product = self.product
            source="user" if dem_path else "py3dep",
            resolution=resolution,
            crs = epsg,
            units = 'meters'
        )

        return dem
