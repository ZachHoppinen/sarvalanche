import numpy as np
import xarray as xr

from sarvalanche.utils.validation import check_rad_degrees

def incidence_angle_weight(
    local_incidence_angle: xr.DataArray | None,
    optimal_range: tuple[float, float] = (35.0, 90.0),
    falloff_width: float = 15.0,
) -> xr.DataArray | float:
    """
    Weight observations based on local incidence angle with maximum weight
    for steep angles (34-90°) and smooth falloff for shallower angles.

    Avalanches are most reliably detected at steeper local incidence angles
    (34-90°) where backscatter contrast is strongest. This function assigns
    maximum weight to this range and smoothly decreases weight for shallower
    angles where detection becomes less reliable.

    Parameters
    ----------
    local_incidence_angle : xr.DataArray | None
        Local incidence angles in degrees (terrain-corrected)
        If None, returns uniform weight of 1.0
    optimal_range : tuple[float, float], default=(34.0, 90.0)
        Range of local incidence angles with maximum weight (degrees)
        Default is [34°, 90°] for reliable avalanche detection
    falloff_width : float, default=15.0
        Standard deviation of Gaussian falloff below optimal range (degrees)
        Controls how quickly weight decreases for angles < 34°

    Returns
    -------
    xr.DataArray | float
        Normalized weights [0, 1] with plateau at optimal_range
        Returns 1.0 if local_incidence_angle is None

    Notes
    -----
    The weighting function:
        w(θ) = 1.0                                    if θ >= 35°
        w(θ) = exp(-(θ - 34)² / (2σ²))               if θ < 35°

    where θ is the local incidence angle and σ is the falloff width (15°).

    This design:
    - Gives full weight (1.0) to angles between 34-90° where avalanche
      backscatter characteristics are most reliable
    - Smoothly reduces weight for shallower angles (< 34°) where detection
      is compromised by layover and foreshortening effects
    - At θ = 35° - σ ≈ 19°, weight drops to ~0.6
    - At θ = 35° - 2σ ≈ 4°, weight drops to ~0.14

    Reference
    ---------
    Bühler, Y., Hafner, E. D., Zweifel, B., Zesiger, M., & Heisig, H. (2021).
    Where are the avalanches? Rapid SPOT6 satellite data acquisition to map an
    extreme avalanche period over the Swiss Alps. The Cryosphere, 15(1), 83-98.
    https://doi.org/10.5194/tc-15-83-2021
    """
    if local_incidence_angle is None:
        return 1.0

    # Check and convert units if needed
    units = check_rad_degrees(local_incidence_angle)
    if units == 'radians':
        local_incidence_angle = np.rad2deg(local_incidence_angle)

    min_angle, max_angle = optimal_range

    # Full weight for angles in optimal range
    w_inc = xr.where(
        local_incidence_angle >= min_angle,
        1.0,
        # Gaussian falloff for angles below optimal range
        np.exp(-((local_incidence_angle - min_angle) ** 2) / (2 * falloff_width ** 2))
    )

    # Ensure weights are in valid range [0, 1]
    w_inc = w_inc.clip(0, 1)

    return w_inc.rename("w_incidence")