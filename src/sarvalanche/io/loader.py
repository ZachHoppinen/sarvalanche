# src/sarvalanche/io/loader.py
from pathlib import Path
from typing import List
import xarray as xr
import h5py
import numpy as np

def load_data(files: List[Path]) -> xr.DataArray:
    """
    Load SAR files into a canonical xarray DataArray.

    Dimensions: (time, y, x)
    
    Parameters
    ----------
    files : List[Path]
        List of SAR file paths.

    Returns
    -------
    xr.DataArray
        Stack of SAR images in canonical form.
    """

    data_list = []
    times = []

    for f in files:
        # TODO: implement sensor-specific loading logic
        with h5py.File(f, "r") as h5:
            # Example: assume dataset called 'data'
            arr = h5["data"][()]
            data_list.append(arr)

            # Example: extract acquisition date
            time_str = h5.attrs.get("acquisition_date", None)
            if time_str is not None:
                times.append(np.datetime64(time_str))
            else:
                times.append(np.datetime64("1970-01-01"))

    # Stack into (time, y, x)
    data_stack = np.stack(data_list, axis=0)
    y = np.arange(data_stack.shape[1])
    x = np.arange(data_stack.shape[2])

    da = xr.DataArray(
        data_stack,
        coords={"time": times, "y": y, "x": x},
        dims=["time", "y", "x"],
        name="sar_data"
    )

    return da
