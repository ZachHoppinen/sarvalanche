# sarvalanche/io/loader/registry.py

from .Sentinel1RTCLoader import Sentinel1RTCLoader
# from .Sentinel1SLCLoader import Sentinel1SLCLoader

LOADERS = {
    ("Sentinel-1", "RTC"): Sentinel1RTCLoader,
}

    # elif dataset in ("s1-slc", "sentinel-1-slc"):
        # return Sentinel1SLCLoader