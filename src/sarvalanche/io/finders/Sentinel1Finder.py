# sarvalanche/io/finder/asf.py

import asf_search as asf
from asf_search.ASFSearchResults import ASFSearchResults

import pandas as pd

from .BaseFinder import BaseFinder

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
        # remove finder-specific filters from kwargs so BaseFinder doesn't see them
        self.path_number = kwargs.pop("path_number", None)
        self.direction = kwargs.pop("direction", None)
        # self.polarization = kwargs.pop("polarization", None)
        self.burst_id = kwargs.pop("burst_id", None)
        self.frame = kwargs.pop("frame", None)

        super().__init__(aoi=aoi, start_date=start_date, stop_date=stop_date, **kwargs)
        self.product_type = product_type

    def query_provider(
        self,
    ) -> ASFSearchResults:
        """
        Query ASF with optional single-value filters.
        Filters can be None or a single value.
        """
        results = asf.geo_search(
            intersectsWith=self.aoi.wkt,
            start=self.start_date,
            end=self.stop_date,
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=self.product_type,
            relativeOrbit=self.path_number,
            relativeBurstID=self.burst_id,
            flightDirection=self.direction,
            frame = self.frame
            # polarization=self.polarization,
        )

        return results

    def find(
        self,
    ) -> list[str]:
        """
        Find ASF URLs with optional single-value filters.

        Filters:
            path_number: int or None
            direction: str ('ASCENDING'/'DESCENDING') or None
            polarization: str ('VV'/'VH') or None
            scene_name: str or None

        Returns:
            List of URLs to matching ASF files.
        """
        # Query provider with optional filters
        asf_results = self.query_provider(
        )

        urls =  asf_results.find_urls()

        # Apply extension filtering only for RTC products
        if self.product_type == asf.PRODUCT_TYPE.RTC:
            urls = self.filter_by_extensions(
                urls,
                extensions=("_VV.tif", "_VH.tif", "_mask.tif"),
            )

        if self.product_type == asf.PRODUCT_TYPE.CSLC:
            urls = self.filter_by_extensions(
                urls,
                extensions=(".h5",),
            )

        # TODO: other filtering for SLC or static products

        return urls
