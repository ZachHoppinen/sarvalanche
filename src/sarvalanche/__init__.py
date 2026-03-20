
import logging

class _SuppressGDALMemoryWarning(logging.Filter):
    def filter(self, record):
        return "'Memory' driver is deprecated" not in record.getMessage()

logging.getLogger('rasterio._env').addFilter(_SuppressGDALMemoryWarning())

from sarvalanche.io.dataset import assemble_dataset  # noqa: E402
from sarvalanche.io.export import export_netcdf  # noqa: E402

__all__ = ["assemble_dataset", "export_netcdf"]
