# sarvalanche/io/finder/asf.py

import asf_search as asf

from .BaseFinder import BaseFinder


class ASFSentinel1Finder(BaseFinder):
    def query_provider(self):
        results = asf.search(
            platform="Sentinel-1",
            start=self.start_date,
            end=self.end_date,
            intersectsWith=self.aoi.wkt,
        )
        return [r.geojson()["properties"]["url"] for r in results]
