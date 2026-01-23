import pytest

import sarvalanche.io as io
from sarvalanche.io.finders.BaseFinder import BaseFinder

class DummyFinder(BaseFinder):
    def query_provider(self):
        # return some test URLs
        return [
            "https://example.com/b.zip",
            "https://example.com/a.zip",
            "https://example.com/a.zip",  # duplicate to test dedup
        ]


def test_finder_deduplicates_and_sorts(aoi, start_date, end_date):
    finder = DummyFinder(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
    )

    urls = finder.find()

    assert urls == [
        "https://example.com/a.zip",
        "https://example.com/b.zip",
    ]

def test_finder_rejects_invalid_urls(aoi, start_date, end_date):
    class BadFinder(BaseFinder):
        def query_provider(self):
            return ["not-a-url"]

    finder = BadFinder(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
    )

    with pytest.raises(ValueError, match="Invalid URL"):
        finder.find()

def test_finder_rejects_invalid_date_range(aoi):
    from datetime import datetime

    finder = DummyFinder(
        aoi=aoi,
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 1, 1),
    )

    with pytest.raises(ValueError):
        finder.find()

def test_normalize_results_strings_dicts_objects():
    class Obj:
        url = "http://example.com/obj.zip"

    f = DummyFinder(aoi=None, start_date="2020-01-01", end_date="2020-01-02")
    results = ["http://example.com/1.zip", {"url": "http://example.com/2.zip"}, Obj()]
    normalized = f.normalize_results(results)
    assert normalized == [
        "http://example.com/1.zip",
        "http://example.com/2.zip",
        "http://example.com/obj.zip",
    ]

def test_filter_results_deduplication():
    f = DummyFinder(aoi=None, start_date="2020-01-01", end_date="2020-01-02")
    urls = ["b", "a", "b"]
    filtered = f.filter_results(urls)
    assert filtered == ["a", "b"]

def test_validate_urls_raises_on_bad():
    f = DummyFinder(aoi=None, start_date="2020-01-01", end_date="2020-01-02")
    with pytest.raises(ValueError):
        f.validate_urls(["ftp://bad.url"])

@pytest.fixture
def urls():
    return [
        "https://example.com/data/S1A_20220101.zip",
        "https://example.com/data/S1B_20220101.zip",
        "https://example.com/data/S1A_20220101.tif",
        "https://example.com/data/readme.txt",
    ]

@pytest.fixture
def dummy_finder(aoi, start_date, end_date):
    return DummyFinder(aoi=aoi, start_date=start_date, end_date=end_date)


def test_filter_by_extensions(urls, dummy_finder):
    # Filter for .zip files
    filtered = dummy_finder.filter_by_extensions(urls, [".zip"])
    assert filtered == [
        "https://example.com/data/S1A_20220101.zip",
        "https://example.com/data/S1B_20220101.zip",
    ]

    # Filter for .tif and .zip files
    filtered = dummy_finder.filter_by_extensions(urls, [".zip", ".tif"])
    assert filtered == [
        "https://example.com/data/S1A_20220101.zip",
        "https://example.com/data/S1B_20220101.zip",
        "https://example.com/data/S1A_20220101.tif",
    ]

    # No match
    filtered = dummy_finder.filter_by_extensions(urls, [".h5"])
    assert filtered == []

def test_filter_by_substring(urls, dummy_finder):
    # Filter for URLs containing "S1A"
    filtered = dummy_finder.filter_by_substring(urls, ["S1A"])
    assert filtered == [
        "https://example.com/data/S1A_20220101.zip",
        "https://example.com/data/S1A_20220101.tif",
    ]

    # Filter for URLs containing "S1B" or "readme"
    filtered = dummy_finder.filter_by_substring(urls, ["S1B", "readme"])
    assert filtered == [
        "https://example.com/data/S1B_20220101.zip",
        "https://example.com/data/readme.txt",
    ]

    # No match
    filtered = dummy_finder.filter_by_substring(urls, ["foobar"])
    assert filtered == []

