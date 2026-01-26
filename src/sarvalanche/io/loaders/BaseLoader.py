# sarvalanche/io/loader/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import xarray as xr
import warnings
from collections import defaultdict
from statistics import median
import re

from sarvalanche.utils import (
    download_urls_parallel,
    _compute_file_checksum,
)


class BaseLoader(ABC):
    """
    Base class for turning data fp into DataArrays.

    Accepts fps -> DataArray
    """

    sensor: str | None = None
    product: str | None = None

    # ---------- public API ----------

    def open(self, path: Path) -> xr.DataArray:
        da = self._open_file(path)
        return self._normalize_dims(da)

    # ---------- subclass hooks ----------

    @abstractmethod
    def _open_file(self, path: Path) -> xr.DataArray:
        """Open a single file into a DataArray"""
        ...

    # ---------- helpers ----------

    def _normalize_dims(self, da: xr.DataArray) -> xr.DataArray:
        dim_map = {
            "latitude": "y",
            "north": "y",
            "y_coord": "y",
            "longitude": "x",
            "east": "x",
            "x_coord": "x",
            "datetime": "time",
            "date": "time"
        }

        rename = {}
        for dim in da.dims:
            canonical = dim_map.get(dim.lower())
            if canonical:
                rename[dim] = canonical

        if rename:
            da = da.rename(rename)

        if not {"y", "x"}.issubset(da.dims):
            raise ValueError(f"Spatial dims missing: {da.dims}")

        return da

    # ---------- optional integrity checks ----------

    def validate_files(
        self,
        files: List[Path],
        *,
        size_outlier_ratio: float = 0.4,
        group_regex: str | None = None,
        checksums: dict[str, str] | None = None,
    ):
        if not files:
            raise ValueError("No files downloaded")

        for f in files:
            if not f.exists():
                raise FileNotFoundError(f)

            size = f.stat().st_size
            if size == 0:
                raise ValueError(f"Empty file: {f}")

            if size < 1024:
                warnings.warn(f"File unusually small: {f}", RuntimeWarning)

        groups = defaultdict(list)
        if group_regex:
            pattern = re.compile(group_regex)
            for f in files:
                m = pattern.search(f.name)
                key = m.group(0) if m else f.suffix
                groups[key].append(f)
        else:
            for f in files:
                groups[f.suffix].append(f)

        for key, group in groups.items():
            if len(group) < 2:
                continue

            sizes = [f.stat().st_size for f in group]
            med = median(sizes)
            for f, size in zip(group, sizes):
                if abs(size - med) / med > size_outlier_ratio:
                    warnings.warn(
                        f"Size outlier in group '{key}': {f.name}",
                        RuntimeWarning,
                    )

        if checksums:
            for f in files:
                expected = checksums.get(f.name)
                if expected and _compute_file_checksum(f) != expected:
                    raise ValueError(f"Checksum mismatch: {f.name}")
