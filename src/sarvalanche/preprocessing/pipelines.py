
import logging
import xarray as xr
from tqdm.auto import tqdm

from sarvalanche.preprocessing.despeckling import denoise_sar_homomorphic
from sarvalanche.utils.constants import pols
from sarvalanche.utils.validation import check_db_linear

log = logging.getLogger(__name__)

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
        log.info(f'Despeckling polarization: {pol}')
        if pol not in ds:
            log.warning(f"Warning: {pol} not found in dataset, skipping")
            continue

        # Validate input is linear
        assert check_db_linear(ds[pol]) == 'linear', f"{pol} must be in linear scale"

        n_times = ds.sizes['time']
        pbar = tqdm(total=n_times, desc=f"Denoising {pol}")

        def denoise_with_progress(da, tv_weight=0.1):
            result = denoise_sar_homomorphic(da, tv_weight=tv_weight)
            pbar.update(1)
            return result

        # Apply denoising across time
        ds[pol] = xr.apply_ufunc(
            denoise_with_progress,
            ds[pol],
            input_core_dims=[['y', 'x']],
            output_core_dims=[['y', 'x']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[ds[pol].dtype],
            kwargs={'tv_weight': tv_weight}
        )

        pbar.close()

    return ds