from .grid import make_reference_grid
from dataclasses import dataclass
import xarray as xr

@dataclass(frozen=True)
class GridSpec:
    bounds: tuple[float, float, float, float]
    crs: str
    resolution: float | tuple[float, float]

    def make_grid(self) -> xr.DataArray:
        return make_reference_grid(
            bounds=self.bounds,
            crs=self.crs,
            resolution=self.resolution,
        )
