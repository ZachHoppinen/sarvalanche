import logging

class _SuppressGDALMemoryWarning(logging.Filter):
    def filter(self, record):
        return "'Memory' driver is deprecated" not in record.getMessage()

logging.getLogger('rasterio._env').addFilter(_SuppressGDALMemoryWarning())
