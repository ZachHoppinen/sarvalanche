import xarray as xr

from sarvalanche.preprocessing.despeckling import denoise_sar_homomorphic
from sarvalanche.utils.constants import pols
from sarvalanche.utils.validation import check_db_linear


def preprocess_rtc(ds, tv_weight=0.1, polarizations=None):
    """
    Preprocess RTC SAR data with homomorphic TV denoising.

    Applies TV denoising in dB space to each polarization independently
    across all timesteps.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing SAR backscatter in linear scale.
        Expected to have polarization variables (e.g., 'VV', 'VH')
        and dimensions (time, y, x).
    tv_weight : float, default 0.1
        TV denoising weight. Higher = more smoothing.
        Typical range: 0.05-0.5 depending on speckle level.
    polarizations : list of str, optional
        List of polarization variables to process.
        If None, uses all polarizations from sarvalanche.utils.constants.pols
        that exist in the dataset.

    Returns
    -------
    ds_denoised : xarray.Dataset
        Dataset with denoised backscatter, preserving all other variables.
    """
    ds = ds.copy()

    # Determine which pols to process
    if polarizations is None:
        polarizations = [p for p in pols if p in ds]

    for pol in polarizations:
        if pol not in ds:
            print(f"Warning: {pol} not found in dataset, skipping")
            continue

        # Validate input is linear
        assert check_db_linear(ds[pol]) == 'linear', f"{pol} must be in linear scale"

        # Apply denoising across time
        ds[pol] = xr.apply_ufunc(
            denoise_sar_homomorphic,
            ds[pol],
            input_core_dims=[['y', 'x']],
            output_core_dims=[['y', 'x']],
            vectorize=True,
            dask='parallelized',  # Enable dask if data is chunked
            kwargs={'tv_weight': tv_weight}
        )

    return ds