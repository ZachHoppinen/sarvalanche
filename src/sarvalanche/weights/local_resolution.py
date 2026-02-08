import xarray as xr

def local_resolution_weights(
        local_resolution : xr.Dataarray,
        resolution_threshold: float = 10):

    local_resolution_weights = 1.0 / (1.0 + (local_resolution / resolution_threshold) ** 3)

    return local_resolution_weights.rename('w_resolution')