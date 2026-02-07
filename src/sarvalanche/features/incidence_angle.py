import numpy as np
import xarray as xr

from sarvalanche.utils.validation import check_rad_degrees

def incidence_angle_weight(
    local_incidence_angle: xr.DataArray | None,
    optimal_angle: float = 55.0,
    angle_width: float = 20.0,
) -> xr.DataArray | float:
    """
    Weight observations based on local incidence angle using a Gaussian centered
    at the optimal angle for avalanche detection.

    Based on Bühler et al. (2021), avalanches appear brightest at local incidence
    angles of 55° ± 20°. This function assigns maximum weight to observations at
    the optimal angle and smoothly decreases weight for angles outside this range.

    Parameters
    ----------
    local_incidence_angle : xr.DataArray | None
        Local incidence angles in degrees (terrain-corrected)
        If None, returns uniform weight of 1.0
    optimal_angle : float, default=55.0
        Optimal local incidence angle for avalanche detection (degrees)
    angle_width : float, default=20.0
        Standard deviation of Gaussian weighting (degrees)
        Defines the range where weights remain high (±1σ = 35-75°)

    Returns
    -------
    xr.DataArray | float
        Normalized weights [0, 1] with peak at optimal_angle
        Returns 1.0 if local_incidence_angle is None

    Notes
    -----
    The Gaussian weighting function:
        w(θ) = exp(-(θ - θ_opt)² / (2σ²))

    where θ is the local incidence angle, θ_opt is the optimal angle (55°),
    and σ is the angle width (20°).

    This heavily weights the 35-75° range where avalanche backscatter
    characteristics are most reliable, while downweighting steep (>75°)
    and shallow (<35°) angles where detection is less reliable.

    Reference
    ---------
    Bühler, Y., Hafner, E. D., Zweifel, B., Zesiger, M., & Heisig, H. (2021).
    Where are the avalanches? Rapid SPOT6 satellite data acquisition to map an
    extreme avalanche period over the Swiss Alps. The Cryosphere, 15(1), 83-98.
    https://doi.org/10.5194/tc-15-83-2021
    """
    if local_incidence_angle is None:
        return 1.0

    # weight by local_incidence_angle
    units = check_rad_degrees(local_incidence_angle)

    if units == 'radians':
        local_incidence_angle = np.rad2deg(local_incidence_angle)

    # Gaussian weighting centered at optimal angle
    w_inc = np.exp(-((local_incidence_angle - optimal_angle) ** 2) /
                   (2 * angle_width ** 2))

    # Ensure weights are in valid range [0, 1]
    w_inc = w_inc.clip(0, 1)

    return w_inc.rename("w_incidence")

