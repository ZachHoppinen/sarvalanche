# sarvalanche/io/loader/registry.py

from .Sentinel1RTCLoader import Sentinel1RTCLoader
# from .Sentinel1SLCLoader import Sentinel1SLCLoader

from sarvalanche.utils.constants import SENTINEL1, OPERA_RTC

LOADERS = {
    (SENTINEL1, OPERA_RTC): Sentinel1RTCLoader,
}

    # elif dataset in ("s1-slc", "sentinel-1-slc"):
        # return Sentinel1SLCLoader