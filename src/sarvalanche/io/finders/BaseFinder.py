from abc import ABC, abstractmethod
from datetime import datetime
import warnings
from shapely.geometry import Polygon

from sarvalanche.utils.validation import validate_aoi, validate_dates

class BaseFinder(ABC):
    def __init__(
        self,
        aoi: Polygon,
        start_date: datetime,
        stop_date: datetime,
    ):
        self.aoi = aoi
        self.start_date = start_date
        self.stop_date = stop_date
    def find(self) -> list[str]:
        self.validate_inputs()
        results = self.query_provider()
        if not results:
            warnings.warn("No results found for the given AOI and date range")
        results = self.normalize_results(results)
        results = self.filter_results(results)
        return self.validate_urls(results)

    def validate_inputs(self):
        validate_aoi(self.aoi)
        validate_dates(self.start_date, self.stop_date)

    @abstractmethod
    def query_provider(self):
        ...

    def normalize_results(self, results, url_field: str = "url") -> list[str]:
        """
        Convert provider-specific results into a list of URL strings.

        Args:
            results: List of provider-specific results (dicts, objects, strings)
            url_field: If result is dict-like, the key to extract URL from

        Returns:
            List of URL strings
        """
        normalized = []

        for r in results:
            # Already a string
            if isinstance(r, str):
                normalized.append(r)
            # Dict-like
            elif isinstance(r, dict):
                if url_field not in r:
                    raise ValueError(f"Result dict missing key '{url_field}': {r}")
                normalized.append(r[url_field])
            # Object with attribute
            elif hasattr(r, url_field):
                normalized.append(getattr(r, url_field))
            else:
                raise TypeError(f"Cannot normalize result: {r}")

        return normalized

    def filter_results(self, urls: list[str], *, deduplicate: bool = True) -> list[str]:
        """
        Filter URLs: remove duplicates, enforce order, optionally filter by other criteria.

        Args:
            urls: List of URL strings
            deduplicate: Remove duplicate URLs

        Returns:
            List of URL strings (unique, sorted)
        """
        if not urls:
            return []

        # Remove duplicates while preserving order
        if deduplicate:
            seen = set()
            unique_urls = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    unique_urls.append(u)
            urls = unique_urls

        # Sort URLs lexicographically
        urls.sort()
        return urls

    def filter_by_extensions(self, urls: list[str], extensions: list[str]) -> list[str]:
        return [u for u in urls if any(u.endswith(ext) for ext in extensions)]

    def filter_by_substring(self, urls: list[str], substrings: list[str]) -> list[str]:
        return [u for u in urls if any(s in u for s in substrings)]


    def validate_urls(self, urls: list[str], *, require_http: bool = True) -> list[str]:
        """
        Validate a list of URLs.

        Args:
            urls: List of strings
            require_http: Enforce that URLs start with 'http' or 'https'

        Returns:
            Cleaned list of URLs

        Raises:
            ValueError: if any URL is invalid
        """
        valid_urls = []

        for u in urls:
            if not isinstance(u, str):
                raise TypeError(f"URL must be a string, got {type(u)}: {u}")
            url = u.strip()
            if require_http and not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError(f"Invalid URL: {url}")
            valid_urls.append(url)

        return valid_urls
