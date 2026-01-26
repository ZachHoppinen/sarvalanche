# sarvalanche/io/sources/base.py

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

import xarray as xr

from sarvalanche.utils import combine_close_images
from sarvalanche.utils.validation import validate_canonical


class BaseDataSource(ABC):
    """
    A DataSource produces canonical xarray objects from user input (AOI, CRS, resolution, dates)
    """

    sensor: str
    product: str

    # ---------- public API ----------

    @abstractmethod
    def load(self, **kwargs) -> xr.DataArray | xr.Dataset:
        ...

    # ---------- shared orchestration helpers ----------

    def _load_files_parallel(
        self,
        loader,
        files: Iterable[Path],
    ) -> list[xr.DataArray]:

        def work(f):
            da = loader.open(f)
            return da

        with ThreadPoolExecutor() as ex:
            return list(ex.map(work, sorted(files)))

    def _stack(self, arrays: list[xr.DataArray]) -> xr.DataArray:
        if len(arrays) == 1:
            return arrays[0]

        if "time" in arrays[0].dims:
            return xr.concat(arrays, dim="time")

        raise ValueError("Multiple arrays but no time dimension")

    def _finalize(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        *,
        urls: list[str] | None = None,
    ) -> xr.DataArray:

        if urls:
            obj.attrs["source_urls"] = urls

        obj.attrs.update(
            sensor=self.sensor,
            product=self.product,
        )

        if isinstance(obj, xr.DataArray):
            validate_canonical(obj)
        elif isinstance(obj, xr.Dataset):
            for da in obj.data_vars.values():
                validate_canonical(da)
        else:
            raise TypeError(f"Expected DataArray or Dataset, got {type(obj)}")

        return obj

    # ---------- optional hooks ----------

    def _parse_time(self, path: Path) -> datetime | None:
        """Override if this source has temporal semantics"""
        return None
