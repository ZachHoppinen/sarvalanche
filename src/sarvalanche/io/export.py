import numpy as np
import pandas as pd

def export_netcdf(ds, filepath, overwrite = True):
    assert filepath.suffix == '.nc'
    ds.attrs['crs'] = str(ds.rio.crs)

    # Reset any MultiIndex before saving
    for coord in ds.coords:
        if isinstance(ds.indexes.get(coord), pd.MultiIndex):
            ds = ds.reset_index(coord)

    # Build encoding - compress all data variables
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4}
        # Remove _FillValue from attributes if it exists (will be set in encoding)
        if '_FillValue' in ds[var].attrs:
            del ds[var].attrs['_FillValue']

        # Set _FillValue in encoding for float types
        if ds[var].dtype.kind == 'f':
            encoding[var]['_FillValue'] = np.nan


    # Explicitly encode scalar string coordinates to prevent mangling
    for coord_name in ['variable', 'band', 'spatial_ref']:
        if coord_name in ds.coords:
            coord = ds.coords[coord_name]
            if coord.dtype.kind in ['U', 'S']:
                # Store as fixed-length string
                encoding[coord_name] = {'dtype': f'S{coord.dtype.itemsize}'}

    if overwrite:
        if filepath.exists():
            filepath.unlink()

    ds.to_netcdf(filepath, encoding=encoding)