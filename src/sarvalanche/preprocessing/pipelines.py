import logging
import xarray as xr
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sarvalanche.preprocessing.despeckling import denoise_sar_homomorphic
from sarvalanche.utils.constants import pols
from sarvalanche.utils.validation import check_db_linear
from sarvalanche.preprocessing.radiometric import linear_to_dB
log = logging.getLogger(__name__)


def _denoise_timestep(args):
    """Top-level function required for pickling with ProcessPoolExecutor."""
    t, arr, tv_weight = args
    result = denoise_sar_homomorphic(arr, tv_weight=tv_weight)
    return t, result


def preprocess_rtc(ds, tv_weight=0.1, polarizations=None, n_workers=4):
    ds = ds.copy()

    if polarizations is None:
        polarizations = [p for p in pols if p in ds]

    for pol in polarizations:
        log.debug(f'Despeckling polarization: {pol}')
        if pol not in ds:
            log.warning(f"{pol} not found in dataset, skipping")
            continue

        assert check_db_linear(ds[pol]) == 'linear', f"{pol} must be in linear scale"

        n_times = ds.sizes['time']
        data = ds[pol].values  # (time, y, x)

        # --- Pre-denoising stats ---
        valid = np.isfinite(data) & (data > 0)
        log.debug(f"  [{pol}] Input  — mean: {data[valid].mean():.4f}  std: {data[valid].std():.4f}  "
                f"min: {data[valid].min():.4f}  max: {data[valid].max():.4f}  "
                f"nan%: {(~valid).mean()*100:.2f}%")
        results = [None] * n_times

        # ProcessPoolExecutor bypasses the GIL for CPU-bound TV denoising,
        # giving 20-40% speedup vs ThreadPoolExecutor.
        # _denoise_timestep must be a top-level function to be picklable.
        tasks = [(t, data[t], tv_weight) for t in range(n_times)]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_denoise_timestep, task): task[0] for task in tasks}
            with tqdm(total=n_times, desc=f"Denoising {pol}") as pbar:
                for future in as_completed(futures):
                    t, result = future.result()
                    results[t] = result
                    pbar.update(1)

        denoised = np.stack(results, axis=0)

        # --- Post-denoising stats ---
        valid_out = np.isfinite(denoised) & (denoised > 0)
        valid_both = valid & valid_out
        log.debug(f"  [{pol}] Output — mean: {denoised[valid_out].mean():.4f}  std: {denoised[valid_out].std():.4f}  "
                f"min: {denoised[valid_out].min():.4f}  max: {denoised[valid_out].max():.4f}  "
                f"nan%: {(~valid_out).mean()*100:.2f}%")
        log.debug(f"  [{pol}] Overall std reduction: "
                f"{(1 - denoised[valid_out].std()/data[valid].std())*100:.1f}%  "
                f"mean absolute change: {np.abs(denoised[valid_both] - data[valid_both]).mean():.4f}")

        # preserve attrs before overwriting
        original_attrs = ds[pol].attrs
        ds[pol] = xr.DataArray(denoised, coords=ds[pol].coords, dims=ds[pol].dims)
        # restore attrs
        ds[pol].attrs = original_attrs

        # we know we are in linear after denoising
        ds[pol] = linear_to_dB(ds[pol])

    return ds