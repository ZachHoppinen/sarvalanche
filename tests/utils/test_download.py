import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from sarvalanche.utils.download import download_urls, download_urls_parallel

@pytest.fixture
def urls():
    return [
        "https://example.com/a.zip",
        "https://example.com/b.zip",
    ]

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

# ----------------------------
# Tests for download_urls
# ----------------------------
def test_download_urls_skips_existing(tmp_dir, urls, mocker):
    # Mock Path.exists() and Path.stat().st_size to simulate existing files
    mock_stat = mocker.MagicMock()
    mock_stat.st_size = 1024

    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
    mock_stat_method = mocker.patch("pathlib.Path.stat", return_value=mock_stat)

    # Also mock download_url so no network call
    mocker.patch("asf_search.download_url")

    files = download_urls(urls, tmp_dir, reprocess=False)

    # Should return Path objects in same order
    expected_files = [tmp_dir / "a.zip", tmp_dir / "b.zip"]
    assert files == expected_files

def test_download_urls_retries(tmp_dir, urls, mocker):
    # Mock download_url to fail once and then succeed
    mock_download = mocker.patch("sarvalanche.utils.download.download_url")
    mock_download.side_effect = [Exception("fail"), None] * len(urls)

    # Mock Path.exists/stats to simulate file not existing
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.stat")

    files = download_urls(urls, tmp_dir, retries=2)

    expected_files = [tmp_dir / "a.zip", tmp_dir / "b.zip"]
    assert files == expected_files
    assert mock_download.call_count == len(urls) * 2  # two attempts per URL

# ----------------------------
# Tests for download_urls_parallel
# ----------------------------
def test_download_urls_parallel_calls_get(tmp_dir, urls, mocker):
    # Patch requests.Session.get
    mock_get = mocker.patch("requests.Session.get")
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"data"]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    files = download_urls_parallel(urls, tmp_dir, max_workers=2, retries=1)

    expected_files = [tmp_dir / "a.zip", tmp_dir / "b.zip"]
    assert sorted(files) == sorted(expected_files)
    assert mock_get.call_count == len(urls)

def test_download_urls_parallel_skips_existing(tmp_dir, urls, mocker):
    # Simulate existing files
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024))

    files = download_urls_parallel(urls, tmp_dir, reprocess=False)

    expected_files = [tmp_dir / "a.zip", tmp_dir / "b.zip"]
    assert files == expected_files
