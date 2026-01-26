from .BaseDataSource import BaseDataSource
from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io.loaders.Sentinel1RTCLoader import Sentinel1RTCLoader
from sarvalanche.grids import GridSpec
from sarvalanche.utils import download_urls_parallel, combine_close_images
from sarvalanche.utils.constants import RTC_FILETYPES
import xarray as xr

class Sentinel1RTCSource(BaseDataSource):
    sensor = "Sentinel-1"
    product = "RTC"

    def __init__(self, *, cache_dir=None):
        self.cache_dir = cache_dir

    def load(
        self,
        *,
        aoi,
        start_date,
        stop_date,
        resolution: float = 30,
        crs: str = "EPSG:4326",
        finder: dict | None = None,
        loader: dict | None = None,
    ):
        finder_kwargs = finder or {}
        loader_kwargs = loader or {}

        reference_grid = GridSpec.make_reference_grid(aoi = aoi, resolution = resolution, crs = crs)

        finder = Sentinel1Finder(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            product_type=self.product,
            **finder_kwargs,
        )

        urls = finder.find()

        fps = download_urls_parallel(urls, out_directory= self.cache_dir)

        loader = Sentinel1RTCLoader(
            **loader_kwargs,
        )

        das = self._load_files_parallel(loader, files = fps)

        ds = xr.Dataset()
        for type in RTC_FILETYPES:
            subtype_fps = [da for da in das if da.name == type]
            da = self._stack(subtype_fps)
            ds[type] = combine_close_images(da)

        return self._finalize(ds)