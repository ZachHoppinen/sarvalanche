# src/sarvalanche/io/__init__.py

from .finders.dispatch import get_finder
from .loader import load_data

__all__ = [
    "get_finder",
    "load_data",
]

def find_data(*, aoi, start_date, stop_date, dataset="auto", **kwargs):
    """
    Convenience wrapper for Finder class.
    Returns list of URLs found for the given AOI and date range.
    """
    finder_cls = get_finder(dataset)
    finder = finder_cls(aoi=aoi, start_date=start_date, stop_date=stop_date, **kwargs)
    return finder.find()

from .loaders.registry import LOADERS

def load_data(
    urls: list[str],
    *,
    sensor: str,
    product: str,
    cache_dir=None,
):
    loader_cls = LOADERS[(sensor, product)]
    loader = loader_cls(cache_dir=cache_dir)
    return loader.load(urls)
