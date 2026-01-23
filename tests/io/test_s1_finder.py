# tests/io/finder/test_asf.py

import pandas as pd
import pytest
from shapely.geometry import box

import asf_search as asf

from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io.finders.asf_utils import subset_asf_search_results

@pytest.fixture
def asf_df():
    return pd.DataFrame(
        {
            "properties.url": [
                "https://example.com/A_VV.tif",
                "https://example.com/B_VH.tif",
            ],
            "properties.additionalUrls": [
                ["https://example.com/A_mask.tif"],
                [],
            ],
            "properties.pathNumber": [42, 43],
            "properties.flightDirection": ["ASCENDING", "DESCENDING"],
            "properties.polarization": ["VV", "VH"],
            "properties.startTime": [
                "2026-01-01T00:00:00",
                "2026-01-01T01:00:00",
            ],
            "properties.stopTime": [
                "2026-01-01T00:10:00",
                "2026-01-01T01:10:00",
            ],
            "properties.sceneName": ["S1A_TEST_001", "S1A_TEST_002"],
            "geometry.coordinates": [
                [[(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]],
                [[(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]],
            ],
        }
    )

def test_get_opera_urls_from_asf_search(asf_df):
    urls = get_opera_urls_from_asf_search(asf_df)

    assert set(urls) == {
        "https://example.com/A_VV.tif",
        "https://example.com/B_VH.tif",
        "https://example.com/A_mask.tif",
    }

def test_subset_by_path_number(asf_df):
    df = subset_asf_search_results(
        asf_df,
        path_numbers=[42],
    )

    assert len(df) == 1
    assert df.iloc[0]["properties.pathNumber"] == 42

def test_subset_by_direction(asf_df):
    df = subset_asf_search_results(
        asf_df,
        direction="ASCENDING",
    )

    assert len(df) == 1
    assert df.iloc[0]["properties.flightDirection"] == "ASCENDING"

def test_subset_by_aoi_wgs(asf_df, aoi_wgs):
    df = subset_asf_search_results(
        asf_df,
        aoi=aoi_wgs,
    )

    # Only first footprint intersects AOI
    assert len(df) == 1

    # Optional but strong assertion: make sure it's the expected scene
    assert df.iloc[0]["properties.url"].endswith("A_VV.tif")

def test_subset_without_aoi_returns_all(asf_df):
    df = subset_asf_search_results(asf_df)

    assert len(df) == len(asf_df)

@pytest.fixture
def fake_asf_results(mocker):
    """Mock ASF results object with geojson() and find_urls()"""
    mock = mocker.MagicMock()
    # mock geojson to return something pd.json_normalize can handle
    mock.geojson.return_value = {
        "features": [
            {"properties": {"url": "https://example.com/A_VV.tif"}},
            {"properties": {"url": "https://example.com/B_VH.tif"}},
        ]
    }
    mock.find_urls.return_value = [
        "https://example.com/A_VV.tif",
        "https://example.com/B_VH.tif",
        "https://example.com/C_mask.tif",
    ]
    return mock

@pytest.mark.parametrize("product_type,expected_filtered", [
    (asf.PRODUCT_TYPE.RTC, True),   # RTC should filter extensions
    ("CSLC", False), # CSLC should skip filter_results
])
def test_find_filters_conditionally(mocker, aoi_wgs, fake_asf_results, product_type, expected_filtered):
    # Patch geo_search to return our fake ASF results
    mocker.patch("sarvalanche.io.finders.Sentinel1Finder.asf.geo_search", return_value=fake_asf_results)

    # Create finder with the given product_type
    finder = Sentinel1Finder(aoi=aoi_wgs, start_date="2024-01-01", stop_date="2024-01-10",
                             product_type=product_type)

    # Patch filter_results to track if it's called
    mock_filter = mocker.patch.object(finder, "filter_by_extensions", wraps=finder.filter_by_extensions)

    urls = finder.find()

    if expected_filtered:
        mock_filter.assert_called_once()
    else:
        mock_filter.assert_not_called()

    # All URLs should still come from find_urls
    assert set(urls).issubset(set(fake_asf_results.find_urls.return_value))

def test_find_subsets_called(mocker, aoi_wgs, fake_asf_results):
    """Test that subset_asf_search_results is called if subset_kwargs are provided"""
    mocker.patch("sarvalanche.io.finders.Sentinel1Finder.asf.geo_search", return_value=fake_asf_results)

    finder = Sentinel1Finder(aoi=aoi_wgs, start_date="2024-01-01", stop_date="2024-01-10")

    # Patch subset_asf_search_results
    mock_subset = mocker.patch("sarvalanche.io.finder.asf_utils.subset_asf_search_results",
                               wraps=lambda df, **kwargs: df)

    finder.find(path_numbers=[1, 2], polarization="VV")

    mock_subset.assert_called_once()
    args, kwargs = mock_subset.call_args
    assert "path_numbers" in kwargs
    assert "polarization" in kwargs
