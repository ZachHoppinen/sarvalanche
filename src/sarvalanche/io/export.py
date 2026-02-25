import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def _sanitize_attrs(attrs: dict) -> dict:
    """Convert attr values to netCDF-safe types."""
    clean = {}
    for k, v in attrs.items():
        if isinstance(v, pd.Timestamp):
            clean[k] = str(v.date())
        elif isinstance(v, np.bool_):
            clean[k] = int(v)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean


def export_netcdf(ds, filepath, overwrite=True):
    assert filepath.suffix == '.nc'
    ds.attrs['crs'] = str(ds.rio.crs)

    # Sanitize dataset-level attrs
    ds.attrs = _sanitize_attrs(ds.attrs)

    # Sanitize all variable and coordinate attrs
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].attrs = _sanitize_attrs(ds[var].attrs)

    # Reset any MultiIndex before saving
    for coord in ds.coords:
        if isinstance(ds.indexes.get(coord), pd.MultiIndex):
            ds = ds.reset_index(coord)

    # Build encoding
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4}
        if '_FillValue' in ds[var].attrs:
            del ds[var].attrs['_FillValue']
        if ds[var].dtype.kind == 'f':
            encoding[var]['_FillValue'] = np.nan

    for coord_name in ['variable', 'band', 'spatial_ref']:
        if coord_name in ds.coords:
            coord = ds.coords[coord_name]
            if coord.dtype.kind in ['U', 'S']:
                encoding[coord_name] = {'dtype': f'S{coord.dtype.itemsize}'}

    if overwrite:
        if filepath.exists():
            filepath.unlink()

    log.info("Writing netCDF: %s", filepath)
    ds.to_netcdf(filepath, encoding=encoding)
    size_mb = filepath.stat().st_size / (1024 ** 2)
    log.info("Wrote %s (%.2f MB)", filepath.name, size_mb)