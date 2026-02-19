# import numpy as np
# import xarray as xr


# def mahalanobis_distance_1d(
#     da: xr.DataArray,
#     avalanche_date: str,
#     prior_window: int = None
# ) -> xr.DataArray:
#     """
#     Calculate 1D Mahalanobis distance for avalanche detection.
#     Uses the logit of da to calculate distance.

#     logit(x) = log(x / (1- x))

#     For each timestep after the avalanche date, computes:
#         d_p = |x_p(t) - μ_p| / σ_p

#     where μ_p and σ_p are the mean and std from the prior period.

#     from: https://arxiv.org/abs/2501.09129

#     Parameters
#     ----------
#     da : xr.DataArray
#         Single polarization backscatter timeseries (time, y, x).
#         Should be preprocessed/denoised and in linear scale.
#     avalanche_date : str
#         Date of avalanche event (e.g., '2021-01-15').
#         Splits data into prior (for computing μ, σ) and post (for detection).
#     prior_window : int, optional
#         Number of timesteps before avalanche_date to use for computing
#         mean/std. If None, uses all available data before avalanche_date.

#     Returns
#     -------
#     distance : xr.DataArray
#         Mahalanobis distance for each pixel at each timestep after
#         avalanche_date. Same shape as input but only post-event times.
#         Values are NaN for timesteps before avalanche_date.
#     """
#     logit_da = np.log(da / (1- da))
#     # Split into prior and post periods
#     prior = logit_da.sel(time=slice(None, avalanche_date))
#     post = logit_da.sel(time=slice(avalanche_date, None))

#     # Optionally limit prior window
#     if prior_window is not None:
#         prior = prior.isel(time=slice(-prior_window, None))

#     # Compute mean and std from prior period (across time)
#     mu = prior.mean(dim='time')
#     sigma = prior.std(dim='time')

#     # Avoid division by zero
#     sigma = sigma.where(sigma > 0, np.nan)

#     # Calculate Mahalanobis distance for post period
#     # while https://arxiv.org/abs/2501.09129 uses the absolute value
#     # we care about whether this is negative or postive
#     # distance = np.abs(post - mu) / sigma
#     distance = (post - mu) / sigma

#     # Preserve time coordinate from post period
#     distance = distance.assign_coords(time=post.time)

#     distance.attrs = {
#         'source': 'sarvalanche',
#         'units': 'standard_deviations',
#         'product': 'mahalanobis_distance',
#         'method': 'mahalanobis_distance',
#     }

#     return distance