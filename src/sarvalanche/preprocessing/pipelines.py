import logging
import xarray as xr
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from sarvalanche.preprocessing.despeckling import denoise_sar_homomorphic
from sarvalanche.utils.constants import pols
from sarvalanche.utils.validation import check_db_linear

log = logging.getLogger(__name__)

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

        def denoise_timestep(t):
            arr = data[t]
            result = denoise_sar_homomorphic(arr, tv_weight=tv_weight)

            return t, result

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(denoise_timestep, t): t for t in range(n_times)}
            with tqdm(total=n_times, desc=f"Denoising {pol}") as pbar:
                for future in as_completed(futures):
                    t, result = future.result()
                    results[t] = result
                    pbar.update(1)

        denoised = np.stack(results, axis=0)

        # --- Post-denoising stats ---
        valid_out = np.isfinite(denoised) & (denoised > 0)
        log.debug(f"  [{pol}] Output — mean: {denoised[valid_out].mean():.4f}  std: {denoised[valid_out].std():.4f}  "
                f"min: {denoised[valid_out].min():.4f}  max: {denoised[valid_out].max():.4f}  "
                f"nan%: {(~valid_out).mean()*100:.2f}%")
        log.debug(f"  [{pol}] Overall std reduction: "
                f"{(1 - denoised[valid_out].std()/data[valid].std())*100:.1f}%  "
                f"mean absolute change: {np.abs(denoised[valid_out] - data[valid_out]).mean():.4f}")

        # preserve attrs before overwriting
        original_attrs = ds[pol].attrs
        ds[pol] = xr.DataArray(denoised, coords=ds[pol].coords, dims=ds[pol].dims)
        # restore attrs
        ds[pol].attrs = original_attrs

    return ds