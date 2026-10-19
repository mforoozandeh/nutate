import numpy as np
import pytest

from nutate import simulate
from nutate.pulses.composite import PRESETS, composite


def test_presets_verbatim():
    assert "10" not in PRESETS
    assert len(PRESETS) == 15
    assert PRESETS["4"] == ((180, 360, 180, 90), (97.2, 291.5, 97.2, 0))
    assert PRESETS["16"] == (
        (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
        (180, 0) * 6,
    )
    assert all(len(f) == len(p) for f, p in PRESETS.values())


def test_composite_180_inverts_on_resonance():
    final = simulate(composite(10e-6, *PRESETS["13"]), [0.0]).final[0]
    np.testing.assert_allclose(final, [0, 0, -1], atol=1e-9)


@pytest.mark.parametrize("flips, phases", [((90, 90), (0,)), ((90.5,), (0,)), ((0,), (0,))])
def test_composite_validation(flips, phases):
    with pytest.raises(ValueError):
        composite(10e-6, flips, phases)


def test_composite_segments():
    segs = composite(10e-6, (90, 180), (0, 120))
    assert [s.n_steps for s in segs] == [90, 180]
    assert segs[0].dt == pytest.approx(10e-6 / 90)
    np.testing.assert_allclose(segs[1].amp, 1.0)
    assert [s.rabi for s in segs] == pytest.approx([25e3, 25e3])
    np.testing.assert_allclose(segs[1].phase, np.deg2rad(120))
