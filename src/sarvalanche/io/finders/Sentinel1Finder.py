# sarvalanche/io/finder/asf.py

import asf_search as asf

import pandas as pd

from .BaseFinder import BaseFinder
from .asf_utils import (
    get_opera_urls_from_asf_search,
    subset_asf_search_results,
)

class Sentinel1Finder(BaseFinder):
    def __init__(
        self,
        *,
        aoi,
        start_date,
        stop_date,
        product_type=asf.PRODUCT_TYPE.RTC,
        **kwargs,
    ):
        super().__init__(aoi=aoi, start_date=start_date, stop_date=stop_date, **kwargs)
        self.product_type = product_type

    def query_provider(self):
        results = asf.geo_search(
            intersectsWith=self.aoi.wkt,
            start=self.start_date,
            end=self.stop_date,
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=self.product_type,
        )

        # Convert to DataFrame once
        df = pd.DataFrame(results.geojson())
        return df

    def find(self, **subset_kwargs):
        df = self.query_provider()

        if subset_kwargs:
            df = subset_asf_search_results(df, aoi=self.aoi, **subset_kwargs)

        urls = get_opera_urls_from_asf_search(df)

        return self.filter_results(
            urls,
            extensions=("_VV.tif", "_VH.tif", "_mask.tif"),
        )