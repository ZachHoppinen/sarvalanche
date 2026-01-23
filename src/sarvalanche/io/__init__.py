# src/sarvalanche/io/__init__.py

from .finders.registry import FINDERS
from .loaders.registry import LOADERS
from .sources.registry import SOURCES

__all__ = [
    "find_data",
    "load_data",
    "load_source",
]

def find_data(*, aoi, start_date, stop_date, source: str, product: str, **kwargs):
    """
    Convenience wrapper for Finder class.
    Returns list of URLs found for the given AOI and date range.
    """
    finder_cls = FINDERS[(source, product)]
    finder = finder_cls(aoi=aoi, start_date=start_date, stop_date=stop_date, **kwargs)
    return finder.find()

def load_data(
    urls: list[str],
    *,
    source: str,
    product: str,
    cache_dir=None,
    substring=None
):
    loader_cls = LOADERS[(source, product)]
    loader = loader_cls(cache_dir=cache_dir, substring = substring)
    return loader.load(urls)

def load_source(
    *,
    source: str,
    product: str,
    cache_dir=None,
    **kwargs,
):
    """
    High-level data access API.

    Examples
    --------
    Sentinel-1:
        load_source(
            source="Sentinel-1",
            product="OPERA-RTC",
            aoi=aoi,
            start_date="2020-01-01",
            stop_date="2020-02-01",
            polarization="VV",
        )

    DEM:
        load_source(
            source="DEM",
            product="3DEP",
            aoi=aoi,
            resolution=30,
        )
    """
    key = (source, product)

    if key not in SOURCES:
        raise KeyError(f"No DataSource registered for {key}")

    source_cls = SOURCES[key]
    src = source_cls(cache_dir=cache_dir)

    return src.load(**kwargs)
