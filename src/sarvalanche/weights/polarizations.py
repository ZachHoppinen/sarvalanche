from sarvalanche.utils.constants import rtc_pol_weights

def get_polarization_weights(polarization: str) -> dict:
    """
    Create polarization weights where VH receives less weight than VV.

    Parameters
    ----------
    polarizations :  str
        polarizations present

    Returns
    -------
    float
        weight to assign for that polarizaiton
    """
    assert polarization in rtc_pol_weights
    return rtc_pol_weights[polarization]