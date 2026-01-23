# sarvalanche/io/loader/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import xarray as xr
import rioxarray
import numpy as np
import hashlib
import re
import warnings
from collections import defaultdict
from statistics import median
from datetime import datetime
from typing import Union

from sarvalanche.utils import download_urls_parallel, _compute_file_checksum, combine_close_images
from sarvalanche.utils.constants import REQUIRED_ATTRS, CANONICAL_DIMS_2D, CANONICAL_DIMS_3D

class BaseLoader(ABC):
    """
    Base class for all data loaders.

    Subclasses should ONLY implement:
      - _open_file()
      - _parse_time()  (if time dimension exists)

    reference_grid is set to first url if not manually set
    target_crs overrides the reference grid is set.
    """

    sensor: str | None = None
    product: str | None = None

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        reference_grid: xr.DataArray | None = None,
        target_crs: str | None = None,
        substring: str | None = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.reference_grid = reference_grid
        self.target_crs = target_crs
        self.substring = substring

    def load(self, urls: list[str]) -> Union[xr.DataArray, xr.Dataset]:
            # Filter by substring if requested
        if self.substring is not None:
            urls = [u for u in urls if self.substring in u]

        files = self._download(urls)
        self._validate_files(files)

        arrays = []
        for f in sorted(files):
            da = self._open_file(f)
            da = self._normalize_dims(da)
            da = self._reproject(da)
            da = self._assign_time(da, f)

            arrays.append(da)

        out = self._stack(arrays)
        out = self._add_attrs(out, urls)
        out = out.sortby('time')
        out = combine_close_images(out, time_tol="2min")
        self._validate_output(out)

        return out

    @abstractmethod
    def _parse_time(self, path: Path) -> datetime:
        ...

    @abstractmethod
    def _open_file(self, path: Path) -> xr.DataArray:
        """
        Open a single file into a DataArray with spatial dims.

    example:
        # def _open_file(self, path):
            # return (
            #     rioxarray.open_rasterio(path, masked=True)
            #     .squeeze("band", drop=True)
            # )
        """
        ...

    def _download(self, urls: list[str]) -> list[Path]:
        if self.cache_dir is None:
            raise ValueError("cache_dir must be set")

        return download_urls_parallel(urls, out_directory = self.cache_dir)

    def _validate_files(
        self,
        files: list[Path],
        *,
        size_outlier_ratio: float = 0.2,
        group_regex: str | None = None,
        checksums: dict[str, str] | None = None,
    ):
        if not files:
            raise ValueError("No files downloaded")

        # ---- basic integrity checks ----
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f)

            size = f.stat().st_size
            if size == 0:
                raise ValueError(f"Empty file: {f}")

            if size < 1024:
                warnings.warn(
                    f"File unusually small (<1KB): {f}",
                    RuntimeWarning,
                )

        # ---- grouping logic ----
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

        # ---- size consistency checks ----
        for key, group in groups.items():
            if len(group) < 2:
                continue

            sizes = [f.stat().st_size for f in group]
            med = median(sizes)

            for f, size in zip(group, sizes):
                if abs(size - med) / med > size_outlier_ratio:
                    warnings.warn(
                        f"Size outlier detected in group '{key}': "
                        f"{f.name} ({size:,} bytes vs median {med:,})",
                        RuntimeWarning,
                    )

        # ---- checksum validation ----
        if checksums:
            for f in files:
                expected = checksums.get(f.name)
                if not expected:
                    continue

                actual = _compute_file_checksum(f)
                if actual != expected:
                    raise ValueError(
                        f"Checksum mismatch for {f.name}: "
                        f"{actual} != {expected}"
                    )

    def _normalize_dims(self, da: xr.DataArray) -> xr.DataArray:
            # Mapping of possible dim names to canonical dims
        dim_map = {
            "latitude": "y",
            "north": "y",
            "y_coord": "y",
            "south": "y",
            "longitude": "x",
            "east": "x",
            "x_coord": "x"
        }

        # Create rename mapping: match lowercase dim names
        rename = {}
        for dim in da.dims:
            canonical = dim_map.get(dim.lower())
            if canonical:
                rename[dim] = canonical

        # Only call rename if there's something to rename
        if rename:
            da = da.rename(rename)

        if not set(("y", "x")).issubset(da.dims):
            raise ValueError(f"Spatial dims missing: {da.dims}")

        return da

    def _assign_time(self, da: xr.DataArray, path: Path) -> xr.DataArray:
        """
        Ensure the DataArray has a time dimension.

        If "time" is already in coords, do nothing.
        Otherwise, use _parse_time(path) to assign a time coordinate.
        If _parse_time returns None, return the array unchanged.
        """
        if "time" in da.coords:
            return da

        t = self._parse_time(path)
        if t is None:
            # No time used, skip
            return da

        # Expand dims with new time coordinate
        return da.expand_dims(time=[t])

    def _stack(self, arrays: list[xr.DataArray]) -> xr.DataArray:
        if len(arrays) == 1:
            return arrays[0]

        if "time" in arrays[0].dims:
            return xr.concat(arrays, dim="time")

        raise ValueError("Multiple arrays but no time dimension")

    def _reproject(self, da: xr.DataArray) -> xr.DataArray:
        if da.rio.crs is None:
            raise ValueError("dataarray must have a valid CRS")

        if self.reference_grid is None:
            # First time: set reference grid from this DA
            self.reference_grid = da

        if self.reference_grid.rio.crs is None:
            raise ValueError("reference_grid must have a valid CRS")

        da = da.rio.reproject_match(self.reference_grid)

        if self.target_crs is not None:
            da = da.rio.reproject(self.target_crs)

        return da

    def _add_attrs(self, da: xr.DataArray, urls: list[str]) -> xr.DataArray:
        attrs = dict(da.attrs)

        attrs.update(
            sensor=self.sensor,
            product=self.product,
            source_urls=urls,
            units = self.units if hasattr(self, 'units') else None
        )

        if da.rio.crs:
            attrs["crs"] = str(da.rio.crs)

        da.attrs = attrs
        return da

    def _validate_output(self, da: xr.DataArray):
        if da.size == 0:
            raise ValueError("DataArray is empty")

        if not set(CANONICAL_DIMS_2D).issubset(da.dims):
            raise ValueError("Output missing spatial dims. Got: {da.dims}")

            # Check all-NaN data
        if np.all(np.isnan(da.values)):
            raise ValueError("DataArray contains only NaNs")

        missing = [
            k for k in REQUIRED_ATTRS
            if k not in da.attrs
        ]
        if missing:
            raise ValueError(f"Missing attrs: {missing}")

