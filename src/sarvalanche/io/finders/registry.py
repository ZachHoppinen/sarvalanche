# sarvalanche/io/finder/dispatch.py

from .Sentinel1Finder import Sentinel1Finder

from sarvalanche.utils.constants import OPERA_RTC

FINDERS = {
    OPERA_RTC: Sentinel1Finder,
}