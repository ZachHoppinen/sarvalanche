from .grid import make_reference_grid
from dataclasses import dataclass
import xarray as xr
import shapely

@dataclass(frozen=True)
class GridSpec:
    aoi: shapely.Polygon
    crs: str
    resolution: float | tuple[float, float]

    def make_grid(self) -> xr.DataArray:
        return make_reference_grid(
            bounds=self.aoi,
            crs=self.crs,
            resolution=self.resolution,
        )
