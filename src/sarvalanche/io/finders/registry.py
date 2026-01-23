# sarvalanche/io/finder/dispatch.py

from .Sentinel1Finder import Sentinel1Finder

from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC


FINDERS = {
    (SENTINEL1, OPERA_RTC): Sentinel1Finder,
}