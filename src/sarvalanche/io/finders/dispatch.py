# sarvalanche/io/finder/dispatch.py

from .Sentinel1Finder import Sentinel1Finder


def get_finder(dataset: str):
    dataset = dataset.lower()
    if dataset in ("sentinel-1", "s1", "auto"):
        return Sentinel1Finder

    raise ValueError(f"Unsupported dataset: {dataset}")