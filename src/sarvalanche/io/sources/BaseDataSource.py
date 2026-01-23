from abc import ABC, abstractmethod

from sarvalanche.utils.validation import validate_canonical

class BaseDataSource(ABC):
    sensor: str
    product: str

    @abstractmethod
    def load(self, **kwargs):
        """Return xr.DataArray or xr.Dataset"""
        ...
