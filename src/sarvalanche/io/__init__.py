# src/sarvalanche/io/__init__.py

from .finders.dispatch import get_finder
from .loader import load_data

__all__ = [
    "get_finder",
    "load_data",
]

def find_data(*, aoi, start_date, end_date, dataset="auto", **kwargs):
    """
    Convenience wrapper for Finder class.
    Returns list of URLs found for the given AOI and date range.
    """
    finder_cls = get_finder(dataset)
    finder = finder_cls(aoi=aoi, start_date=start_date, end_date=end_date, **kwargs)
    return finder.find()
