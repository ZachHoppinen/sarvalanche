from datetime import datetime
from shapely.geometry import box

import sarvalanche.io as io


def test_find_data_returns_urls(mocker, aoi, start_date, end_date):
    fake_results = [
        {"url": "https://example.com/s1_001.zip"},
        {"url": "https://example.com/s1_002.zip"},
    ]

    mock_search = mocker.patch(
        "sarvalanche.io.search.asf_search.search",
        return_value=fake_results,
    )

    urls = io.find_data(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor="Sentinel-1",
    )

    assert isinstance(urls, list)
    assert len(urls) == 2
    assert urls[0].endswith(".zip")

    mock_search.assert_called_once()
