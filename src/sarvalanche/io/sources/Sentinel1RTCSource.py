from .BaseDataSource import BaseDataSource
from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io.loaders.Sentinel1RTCLoader import Sentinel1RTCLoader

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

        return loader.load(urls)
