from .Sentinel1RTCSource import Sentinel1RTCSource
from .DemSource import DemSource
from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

SOURCES = {
    (SENTINEL1, OPERA_RTC): Sentinel1RTCSource,
    ("3DEP", "DEM"): DemSource,
}
