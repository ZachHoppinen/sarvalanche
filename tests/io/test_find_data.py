
from unittest.mock import patch, MagicMock
import pytest

import asf_search as asf
from shapely.geometry import box

from sarvalanche.io.find_urls import (
    find_asf_urls
)

@patch("sarvalanche.utils.validation.validate_urls", side_effect=lambda urls: urls)
@patch("asf_search.geo_search")
def test_find_asf_urls_rtc_filter(mock_geo_search, mock_validate, mock_asf_urls, aoi_wgs):
    # Setup mock
    mock_results = MagicMock()
    mock_results.find_urls.return_value = mock_asf_urls
    mock_geo_search.return_value = mock_results

    urls = find_asf_urls(aoi_wgs, "2020-01-01", "2020-01-31", product_type=asf.PRODUCT_TYPE.RTC)

    # Only RTC extensions (_VV.tif, _VH.tif, _mask.tif)
    expected = [
        "https://example.com/file_VV.tif",
        "https://example.com/file_VH.tif",
        "https://example.com/file_mask.tif",
    ]
    assert urls == expected

@patch("sarvalanche.utils.validation.validate_urls", side_effect=lambda urls: urls)
@patch("asf_search.geo_search")
def test_find_asf_urls_cslc_filter(mock_geo_search, mock_validate, mock_asf_urls, aoi_wgs):
    mock_results = MagicMock()
    mock_results.find_urls.return_value = mock_asf_urls
    mock_geo_search.return_value = mock_results

    urls = find_asf_urls(aoi_wgs, "2020-01-01", "2020-01-31", product_type=asf.PRODUCT_TYPE.CSLC)

    # Only .h5
    expected = ["https://example.com/file.h5"]
    assert urls == expected

@patch("sarvalanche.utils.validation.validate_urls", side_effect=lambda urls: urls)
@patch("asf_search.geo_search")
def test_find_asf_urls_empty(mock_geo_search, mock_validate, aoi_wgs):
    mock_results = MagicMock()
    mock_results.find_urls.return_value = []
    mock_geo_search.return_value = mock_results

    with pytest.raises(ValueError):
        urls = find_asf_urls(aoi_wgs, "2020-01-01", "2020-01-31")

@patch("sarvalanche.utils.validation.validate_urls", side_effect=lambda urls: urls)
@patch("asf_search.geo_search")
def test_find_asf_urls_passes_parameters(mock_geo_search, mock_validate, mock_asf_urls, aoi_wgs):
    mock_results = MagicMock()
    mock_results.find_urls.return_value = mock_asf_urls
    mock_geo_search.return_value = mock_results

    find_asf_urls(
        aoi_wgs,
        "2020-01-01",
        "2020-01-31",
        platform=asf.PLATFORM.SENTINEL1,
        path_number=5,
        burst_id=2,
        direction="ASCENDING",
        frame="0010"
    )

    # Check geo_search called with correct args
    mock_geo_search.assert_called_once()
    args, kwargs = mock_geo_search.call_args
    assert kwargs["intersectsWith"] == aoi_wgs.wkt
    assert kwargs["start"] == "2020-01-01"
    assert kwargs["end"] == "2020-01-31"
    assert kwargs["platform"] == asf.PLATFORM.SENTINEL1
    assert kwargs["processingLevel"] == asf.PRODUCT_TYPE.RTC
    assert kwargs["relativeOrbit"] == 5
    assert kwargs["relativeBurstID"] == 2
    assert kwargs["flightDirection"] == "ASCENDING"
    assert kwargs["frame"] == "0010"

@pytest.mark.network
@pytest.mark.slow
def test_find_data_sentinel1_opera_rtc_real_asf():
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

    urls = find_asf_urls(
        aoi=aoi,
        start_date=start_date,
        stop_date=end_date,
        platform=asf.PLATFORM.SENTINEL1,
        product_type= asf.PRODUCT_TYPE.RTC
    )

    # --- Basic sanity checks ---
    assert isinstance(urls, list)
    assert len(urls) == 6

    # --- Assert expected file types ---
    assert all(u.endswith(".tif") for u in urls)

    # --- Assert OPERA RTC path ---
    assert all("OPERA_L2_RTC-S1" in u for u in urls)
    
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
