# tests/io/finder/test_asf.py

import pandas as pd
import pytest
from shapely.geometry import box

import asf_search as asf

from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io.finders.asf_utils import (
    get_opera_urls_from_asf_search,
    subset_asf_search_results,
)

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

# def test_subset_by_aoi_projected_raises(asf_df, aoi_projected):
#     with pytest.raises(ValueError, match="EPSG:4326"):
#         subset_asf_search_results(
#             asf_df,
#             aoi=aoi_projected,
#         )

def test_subset_without_aoi_returns_all(asf_df):
    df = subset_asf_search_results(asf_df)

    assert len(df) == len(asf_df)


def test_find_returns_filtered_urls(mocker, aoi_wgs, start_date, end_date, asf_df):
    # Mock ASF search
    mocker.patch(
        "sarvalanche.io.finders.Sentinel1Finder.asf.geo_search",
        return_value="FAKE_RESULTS",
    )

    # Mock dataframe conversion
    mocker.patch(
        "sarvalanche.io.finders.Sentinel1Finder.asf.results_to_dataframe",
        return_value=asf_df,
    )

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date=start_date,
        stop_date=end_date,
    )

    urls = finder.find()

    assert urls == sorted(
        [
            "https://example.com/A_VV.tif",
            "https://example.com/B_VH.tif",
            "https://example.com/A_mask.tif",
        ]
    )

def test_product_type_passed_to_asf(mocker, aoi_wgs, start_date, end_date, asf_df):
    geo_search = mocker.patch(
        "sarvalanche.io.finders.Sentinel1Finder.asf.geo_search",
        return_value="FAKE_RESULTS",
    )

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date=start_date,
        stop_date=end_date,
        product_type=asf.PRODUCT_TYPE.CSLC,
    )

    finder.find()

    geo_search.assert_called_once()
    assert geo_search.call_args.kwargs["processingLevel"] == asf.PRODUCT_TYPE.CSLC
