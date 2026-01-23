# sarvalanche/io/finder/dispatch.py

from .Sentinel1Finder import ASFSentinel1Finder


def get_finder(dataset: str):
    dataset = dataset.lower()
    if dataset in ("sentinel-1", "s1", "auto"):
        return ASFSentinel1Finder

    raise ValueError(f"Unsupported dataset: {dataset}")