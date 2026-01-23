import warnings
import pytest
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, box

from sarvalanche.utils.validation import (
    validate_dates,
    validate_aoi,
    within_conus,
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
