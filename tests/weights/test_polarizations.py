import pytest

from sarvalanche.weights.polarizations import get_polarization_weights


@pytest.mark.parametrize(
    "polarization, expected_weight",
    [
        ("VV", 1.0),
        ("VH", 0.7),
    ],
)
def test_known_polarization_weights(polarization, expected_weight):
    assert get_polarization_weights(polarization) == pytest.approx(expected_weight)


def test_vv_has_higher_weight_than_vh():
    assert get_polarization_weights("VV") > get_polarization_weights("VH")


def test_invalid_polarization_raises():
    with pytest.raises(AssertionError):
        get_polarization_weights("HH")


def test_invalid_polarization_lowercase_raises():
    with pytest.raises(AssertionError):
        get_polarization_weights("vv")
