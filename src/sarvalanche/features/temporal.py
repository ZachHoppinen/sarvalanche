
import numpy as np
import xarray as xr

def temporal_weights(
    diffs: xr.DataArray,
    tau_days: float,
    pair_dim: str = "pair",
) -> xr.DataArray:
    """
    Exponential temporal weighting for backscatter pairs.
    Shorter intervals get higher weight.
    """
    dt_seconds = (diffs.t_end - diffs.t_start).astype("timedelta64[s]")
    dt_days = (dt_seconds / (24 * 3600)).astype(float)

    w_temporal = np.exp(-dt_days / tau_days)

    return xr.DataArray(
        w_temporal,
        dims=[pair_dim],
        coords={pair_dim: diffs[pair_dim]},
        name="w_temporal",
    )
