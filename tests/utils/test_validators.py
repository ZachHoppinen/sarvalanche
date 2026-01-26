import warnings
import pytest
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, box
from pyproj import CRS
from pyproj.exceptions import CRSError

from sarvalanche.utils.validation import (
    validate_dates,
    validate_aoi,
    within_conus,
    validate_crs,
    validate_resolution,
    validate_urls
)

def test_validate_dates_basic():
    start, end = validate_dates("2020-01-01", "2020-02-01")

    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)
    assert start < end

def test_validate_dates_start_after_end():
    with pytest.raises(ValueError, match="start_date must be earlier"):
        validate_dates("2020-02-01", "2020-01-01")

def test_validate_dates_before_2014():
    with pytest.raises(ValueError, match="2014"):
        validate_dates("2010-01-01", "2015-01-01")

def test_validate_dates_future():
    future = pd.Timestamp.now() + pd.Timedelta(days=10)
    with pytest.raises(ValueError, match="future"):
        validate_dates("2020-01-01", future)

def test_validate_dates_timezone_mismatch():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-02-01")
    with pytest.raises(ValueError, match="same timezone"):
        validate_dates(start, end)

def test_validate_dates_s1b_warning():
    with pytest.warns(UserWarning, match="Sentinel-1B outage"):
        validate_dates("2022-01-01", "2022-02-01")

def test_validate_aoi_bbox_list():
    aoi = validate_aoi([-120, 30, -110, 40])
    assert isinstance(aoi, Polygon)
    assert aoi.bounds == (-120, 30, -110, 40)

def test_validate_aoi_reversed_bounds():
    aoi = validate_aoi([-110, 40, -120, 30])
    assert aoi.bounds == (-120, 30, -110, 40)

def test_validate_aoi_point_list():
    aoi = validate_aoi([-120, 35])
    assert isinstance(aoi, Point)
    assert aoi.x == -120
    assert aoi.y == 35

@pytest.mark.parametrize(
    "aoi_dict",
    [
        {"xmin": -120, "ymin": 30, "xmax": -110, "ymax": 40},
        {"west": -120, "south": 30, "east": -110, "north": 40},
        {"minx": -120, "miny": 30, "maxx": -110, "maxy": 40},
    ],
)
def test_validate_aoi_dicts(aoi_dict):
    aoi = validate_aoi(aoi_dict)
    assert isinstance(aoi, Polygon)
    assert aoi.bounds == (-120, 30, -110, 40)

def test_validate_aoi_existing_geometry():
    geom = box(-120, 30, -110, 40)
    out = validate_aoi(geom)
    assert out is geom

def test_validate_aoi_invalid():
    with pytest.raises(TypeError, match="AOI must be geometry"):
        validate_aoi("not an aoi")

def test_within_conus_true():
    assert within_conus([-120, 35, -110, 45]) is True

def test_within_conus_false():
    assert within_conus([-160, 60, -150, 65]) is False

def test_within_conus_partial_overlap():
    assert within_conus([-130, 40, -120, 45]) is True

def test_validate_crs_from_pyproj_crs():
    crs_obj = CRS.from_epsg(4326)
    out = validate_crs(crs_obj)
    assert isinstance(out, CRS)
    assert out.to_epsg() == 4326

def test_validate_crs_from_epsg_int():
    out = validate_crs(32633)
    assert isinstance(out, CRS)
    assert out.to_epsg() == 32633

def test_validate_crs_from_epsg_string():
    out = validate_crs("EPSG:3857")
    assert isinstance(out, CRS)
    assert out.to_epsg() == 3857

def test_validate_crs_from_proj_string():
    out = validate_crs("EPSG:4326")
    assert isinstance(out, CRS)
    assert out.is_geographic

def test_validate_crs_invalid_string():
    with pytest.raises(ValueError):
        validate_crs("INVALID_CRS_STRING")

def test_validate_crs_invalid_epsg():
    with pytest.raises(ValueError):
        validate_crs(999999)  # non-existent EPSG

def test_validate_crs_wrong_type():
    with pytest.raises(TypeError):
        validate_crs([4326])  # list is invalid

def test_validate_crs_invalid_pyproj_crs():
    # Create an invalid CRS
    with pytest.raises(CRSError):
        crs = CRS.from_wkt("")  # invalid CRS

def test_single_float_resolution():
    res = 10
    xres, yres = validate_resolution(res)
    assert xres == 10
    assert yres == 10

def test_tuple_resolution():
    res = (5, 20)
    xres, yres = validate_resolution(res)
    assert xres == 5
    assert yres == 20

def test_list_resolution():
    res = [7, 14]
    xres, yres = validate_resolution(res)
    assert xres == 7
    assert yres == 14

def test_zero_resolution():
    with pytest.raises(ValueError):
        validate_resolution(0)
    with pytest.raises(ValueError):
        validate_resolution((10, 0))

def test_negative_resolution():
    with pytest.raises(ValueError):
        validate_resolution(-5)
    with pytest.raises(ValueError):
        validate_resolution((5, -2))

def test_invalid_type():
    with pytest.raises(TypeError):
        validate_resolution("10")
    with pytest.raises(ValueError):
        validate_resolution([1, 2, 3])  # length != 2

def test_tuple_wrong_length():
    with pytest.raises(ValueError):
        validate_resolution((1,))
    with pytest.raises(ValueError):
        validate_resolution((1, 2, 3))

def test_validate_resolution_with_projected_crs_ok():
    crs = CRS.from_epsg(32633)  # UTM zone 33N, projected meters
    res = 10.0
    xres, yres = validate_resolution(res, crs=crs)
    assert xres == 10.0 and yres == 10.0

def test_validate_resolution_with_projected_crs_too_small():
    crs = CRS.from_epsg(32633)  # projected meters
    with pytest.raises(ValueError, match="unrealistically small"):
        validate_resolution(0.001, crs=crs)

def test_validate_resolution_with_geographic_crs_ok():
    crs = CRS.from_epsg(4326)  # geographic degrees
    res = 0.01
    xres, yres = validate_resolution(res, crs=crs)
    assert xres == 0.01 and yres == 0.01

def test_validate_resolution_with_geographic_crs_too_small():
    crs = CRS.from_epsg(4326)
    with pytest.raises(ValueError, match="unrealistically small"):
        validate_resolution(1e-6, crs=crs)

def test_validate_resolution_invalid_crs_type():
    # Passing something that is not a pyproj CRS
    with pytest.raises(TypeError, match="CRS must be a pyproj.CRS object"):
        validate_resolution(10, crs="EPSG:4326")

def test_validate_resolution_without_crs_still_works():
    # Ensure original functionality unchanged
    xres, yres = validate_resolution(5)
    assert xres == 5 and yres == 5
    xres, yres = validate_resolution((3, 7))
    assert xres == 3 and yres == 7


def test_validate_urls_valid_http():
    urls = ["http://example.com", "https://example.org"]
    result = validate_urls(urls)
    assert result == urls

def test_validate_urls_strip_whitespace():
    urls = ["  http://example.com  ", "\thttps://example.org\n"]
    result = validate_urls(urls)
    assert result == ["http://example.com", "https://example.org"]

def test_validate_urls_require_http_default_raises():
    urls = ["ftp://example.com", "example.org"]
    with pytest.raises(ValueError):
        validate_urls(urls)

def test_validate_urls_require_http_false_allows():
    urls = ["ftp://example.com", "example.org"]
    result = validate_urls(urls, require_http=False)
    assert result == urls

def test_validate_urls_non_string_raises():
    urls = ["http://example.com", 123, None]
    with pytest.raises(TypeError):
        validate_urls(urls)

def test_validate_urls_mixed_valid_and_invalid_http():
    urls = ["http://example.com", "ftp://example.org"]
    with pytest.raises(ValueError):
        validate_urls(urls)

def test_validate_urls_empty_list_returns_empty():
    with pytest.raises(ValueError):
        result = validate_urls([])
