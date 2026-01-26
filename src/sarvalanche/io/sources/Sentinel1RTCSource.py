from .BaseDataSource import BaseDataSource
from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io.loaders.Sentinel1RTCLoader import Sentinel1RTCLoader

from sarvalanche.utils.projections import get_aoi_utm_bounds

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
        finder: dict | None = None,
        loader: dict | None = None,
    ):
        finder_kwargs = finder or {}
        loader_kwargs = loader or {}

        finder = Sentinel1Finder(
            aoi=aoi,
            start_date=start_date,
            stop_date=stop_date,
            product_type=self.product,
            **finder_kwargs,
        )

        urls = finder.find()

        loader = Sentinel1RTCLoader(
            cache_dir=self.cache_dir,
            **loader_kwargs,
        )

        da = loader.load(urls)
        utm_bounds = get_aoi_utm_bounds(aoi, da.rio.crs)
        print(utm_bounds)
        da = da.rio.clip_box(*utm_bounds).rio.pad_box(*utm_bounds)
        return da