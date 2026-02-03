import xarray as xr

def incidence_angle_weight(
    local_incidence_angle: xr.DataArray | None,
    incidence_power: float = 1.0,
) -> xr.DataArray | float:
    """
    Normalize and weight local incidence angle.
    """
    if local_incidence_angle is None:
        return 1.0

    min_inc = local_incidence_angle.min()
    max_inc = local_incidence_angle.max()

    w_inc = (local_incidence_angle - min_inc) / (max_inc - min_inc + 1e-6)
    w_inc = (w_inc ** incidence_power).clip(0, 1)

    return w_inc.rename("w_incidence")