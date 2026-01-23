# sarvalanche/io/finder/asf.py

import asf_search as asf

import pandas as pd

from .BaseFinder import BaseFinder
from .asf_utils import subset_asf_search_results

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

        return results

    def find(self, **subset_kwargs):
        asf_results = self.query_provider()

        results_df = pd.json_normalize(asf_results.geojson(), record_path = ['features'])

        if subset_kwargs:
            df = subset_asf_search_results(results_df, aoi=self.aoi, **subset_kwargs)

        urls =  asf_results.find_urls()

        # Apply extension filtering only for RTC products
        if self.product_type == asf.PRODUCT_TYPE.RTC:
            urls = self.filter_by_extensions(
                urls,
                extensions=("_VV.tif", "_VH.tif", "_mask.tif"),
            )

        # TODO: other filtering for SLC or static products

        return urls
