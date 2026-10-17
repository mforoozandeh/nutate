import pytest

from nutate._format import si_scale


@pytest.mark.parametrize(
    "x, unit, expected",
    [
        (15e3, "Hz", (1e-3, "kHz")),
        (2e-3, "s", (1e3, "ms")),
        (5e-7, "s", (1e9, "ns")),
        (2e-5, "s", (1e6, "µs")),
        (3e8, "Hz", (1e-6, "MHz")),
        (0.0, "Hz", (1.0, "Hz")),
    ],
)
def test_si_scale(x, unit, expected):
    f, u = si_scale(x, unit)
    assert u == expected[1]
    assert f == pytest.approx(expected[0])
