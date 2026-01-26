# tests/io/finder/test_asf.py

import pandas as pd
from unittest.mock import MagicMock

import pytest
from shapely.geometry import box

import asf_search as asf

from sarvalanche.io.finders.Sentinel1Finder import Sentinel1Finder
from sarvalanche.io import find_data

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



@pytest.fixture
def fake_asf_results(mocker):
    """Mock ASF results object with geojson() and find_urls()"""
    mock = mocker.MagicMock()
    # mock geojson to return something pd.json_normalize can handle
    mock.geojson.return_value = {
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]],
            },
            "properties": {
                "url": "https://example.com/A_VV.tif",
                "additionalUrls": ["https://example.com/A_mask.tif"],
                "pathNumber": 1,
                "polarization": "VV",
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]],
            },
            "properties": {
                "url": "https://example.com/B_VH.tif",
                "additionalUrls": [],
                "pathNumber": 2,
                "polarization": "VH",
            },
        },
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
    ("CSLC", True), # CSLC should skip filter_results
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

@pytest.fixture
def aoi_wgs():
    # small box AOI
    return box(-155.5, 19.9, -155.4, 20.0)


@pytest.fixture
def fake_asf_results():
    mock = MagicMock()
    mock.find_urls.return_value = [
        "https://example.com/A_VV.tif",
        "https://example.com/A_VH.tif",
        "https://example.com/A_mask.tif",
        "https://example.com/B_VV.tif",
    ]
    return mock


def test_init_with_filters(aoi_wgs):
    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date="2020-01-01",
        stop_date="2020-01-10",
        path_number=42,
        direction="ASCENDING",
        burst_id=7,
        frame=3,
    )

    assert finder.path_number == 42
    assert finder.direction == "ASCENDING"
    assert finder.burst_id == 7
    assert finder.frame == 3


def test_query_provider_calls_geo_search(mocker, aoi_wgs):
    # patch asf.geo_search
    geo_patch = mocker.patch("asf_search.geo_search", return_value=MagicMock())

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date="2020-01-01",
        stop_date="2020-01-10",
        path_number=87,
        direction="DESCENDING",
        burst_id=5,
        frame=2,
    )

    results = finder.query_provider()
    geo_patch.assert_called_once()

    # check that correct arguments passed
    called_kwargs = geo_patch.call_args.kwargs
    assert called_kwargs["intersectsWith"] == aoi_wgs.wkt
    assert called_kwargs["relativeOrbit"] == 87
    assert called_kwargs["flightDirection"] == "DESCENDING"
    assert called_kwargs["relativeBurstID"] == 5
    assert called_kwargs["frame"] == 2
    assert called_kwargs["processingLevel"] == asf.PRODUCT_TYPE.RTC
    assert results == geo_patch.return_value


def test_find_rtc_extension_filtering(mocker, aoi_wgs, fake_asf_results):
    # patch query_provider to return our fake ASFSearchResults
    mocker.patch.object(Sentinel1Finder, "query_provider", return_value=fake_asf_results)

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date="2020-01-01",
        stop_date="2020-01-10",
        product_type=asf.PRODUCT_TYPE.RTC,
    )

    urls = finder.find()
    # RTC should filter to _VV, _VH, _mask
    for u in urls:
        assert u.endswith(("_VV.tif", "_VH.tif", "_mask.tif"))

    # All URLs from fake_asf_results.find_urls that match
    assert len(urls) == 4  # 3 matching


def test_find_cslc_extension_filtering(mocker, aoi_wgs, fake_asf_results):
    fake_asf_results.find_urls.return_value = [
        "file1.h5",
        "file2.h5",
        "file3.tif",
    ]

    mocker.patch.object(Sentinel1Finder, "query_provider", return_value=fake_asf_results)

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date="2020-01-01",
        stop_date="2020-01-10",
        product_type=asf.PRODUCT_TYPE.CSLC,
    )

    urls = finder.find()
    # CSLC should only keep .h5
    assert all(u.endswith(".h5") for u in urls)
    assert len(urls) == 2


def test_find_no_filters_applied(mocker, aoi_wgs, fake_asf_results):
    # product_type that does not trigger filtering
    mocker.patch.object(Sentinel1Finder, "query_provider", return_value=fake_asf_results)

    finder = Sentinel1Finder(
        aoi=aoi_wgs,
        start_date="2020-01-01",
        stop_date="2020-01-10",
        product_type="SLC",
    )

    urls = finder.find()
    # all URLs returned since no filtering applied
    assert urls == fake_asf_results.find_urls()

@pytest.mark.network
@pytest.mark.slow
def test_find_data_sentinel1_cslc_real_asf():
    """
    Real ASF network test for Sentinel-1 OPERA RTC products.

    This test:
    - Hits ASF directly
    - Uses a small AOI + narrow time window
    - Asserts known OPERA RTC outputs
    """

    aoi = box(-155.5, 19.9, -155.4, 20.0)
    start_date = "2020-01-01"
    end_date = "2020-01-04"

    urls = find_data(
        aoi=aoi,
        start_date=start_date,
        stop_date=end_date,
        # dataset="Sentinel-1",
        source = 'SENTINEL-1',
        product = 'OPERA-RTC'
    )

    # --- Basic sanity checks ---
    assert isinstance(urls, list)
    assert len(urls) == 6

    # --- Assert expected file types ---
    assert all(
        u.endswith((".tif",))
        for u in urls
    )

    # --- Assert OPERA RTC path ---
    assert all(
        "OPERA_L2_RTC-S1" in u
        for u in urls
    )

    # --- Assert exact expected set (order-insensitive) ---
    expected = {
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0_VH.tif',
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0_VV.tif',
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185679-IW2_20200101T161622Z_20250911T203016Z_S1A_30_v1.0_mask.tif',
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0_VH.tif',
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0_VV.tif',
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0/OPERA_L2_RTC-S1_T087-185680-IW2_20200101T161625Z_20250911T203016Z_S1A_30_v1.0_mask.tif'
    }

    assert set(urls) == expected