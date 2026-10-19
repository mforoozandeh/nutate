import numpy as np
import pytest

from nutate import Segment, delay, simulate, waveform


def test_segment_properties():
    s = Segment("x", amp=np.full(4, 0.5), phase=np.zeros(4), dt=1e-6, rabi=10e3)
    assert s.n_steps == 4
    assert s.duration == pytest.approx(4e-6)
    np.testing.assert_allclose(s.drive, 5e3)  # Hz: amp · rabi


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"amp": np.ones(3), "phase": np.zeros(4), "dt": 1e-6}, "phase"),
        ({"amp": -np.ones(3), "phase": np.zeros(3), "dt": 1e-6}, "amp"),
        ({"amp": np.ones(3), "phase": np.zeros(3), "dt": 0.0}, "dt"),
        ({"amp": np.array([1.0, np.nan]), "phase": np.zeros(2), "dt": 1e-6}, "amp"),
        ({"amp": np.ones(0), "phase": np.zeros(0), "dt": 1e-6}, "amp"),
        ({"amp": np.full(3, 2.0), "phase": np.zeros(3), "dt": 1e-6}, "normalised"),
        ({"amp": np.ones(3), "phase": np.zeros(3), "dt": 1e-6, "rabi": -1.0}, "rabi"),
        ({"amp": np.ones(3), "phase": np.zeros(3), "dt": 1e-6, "rabi": np.nan}, "rabi"),
    ],
)
def test_segment_validation(kwargs, field):
    with pytest.raises(ValueError, match=field):
        Segment("x", **{"rabi": 1e3, **kwargs})


def test_delay_is_zero_amplitude():
    s = delay(1e-3, 50)
    assert s.n_steps == 50
    assert s.dt == pytest.approx(2e-5)
    assert not s.amp.any()


def test_with_rabi_sets_rabi_only():
    c = waveform(np.linspace(0, 1, 5), np.ones(5), duration=1e-6, rabi=10e3)
    scaled = c.with_rabi(1234.0)
    assert scaled.rabi == 1234.0
    np.testing.assert_array_equal(scaled.amp, c.amp)
    np.testing.assert_array_equal(scaled.phase, c.phase)
    assert scaled.dt == c.dt
    assert scaled.name == c.name


def test_with_rabi_validates():
    with pytest.raises(ValueError, match="rabi"):
        waveform([1.0], [0.0], duration=1e-6, rabi=1e3).with_rabi(-1.0)


def test_waveform_scales_by_rabi_without_rescaling():
    s = waveform([0.5, 1.05], [0.0, np.pi / 2], duration=2e-6, rabi=1e4, name="w")
    np.testing.assert_allclose(s.amp, [0.5, 1.05])  # overshoot above 1 kept, not clipped
    assert s.rabi == 1e4
    np.testing.assert_allclose(s.phase, [0.0, np.pi / 2])
    assert s.dt == pytest.approx(1e-6)
    assert s.name == "w"


def test_waveform_t90_gives_rabi():  # ν1 = 1/(4·t90): a unit waveform lasting t90 is a 90°
    s = waveform(np.ones(10), np.zeros(10), duration=10e-6, t90=10e-6)
    assert s.rabi == pytest.approx(25e3)
    np.testing.assert_allclose(simulate([s], [0.0]).final[0], [0, -1, 0], atol=1e-9)


def test_waveform_iq_puts_q_along_y():
    s = waveform(i=np.zeros(10), q=np.ones(10), duration=10e-6, t90=10e-6)
    np.testing.assert_allclose(simulate([s], [0.0]).final[0], [1, 0, 0], atol=1e-9)


def test_waveform_iq_amplitude_and_phase():
    s = waveform(i=[1.0, 0.0, -0.6], q=[0.0, 0.5, 0.8], duration=3e-6, rabi=1e4)
    np.testing.assert_allclose(s.amp, [1.0, 0.5, 1.0])
    np.testing.assert_allclose(s.phase, [0.0, np.pi / 2, np.arctan2(0.8, -0.6)])


def test_waveform_phase_deg():
    s = waveform([1.0, 1.0], phase_deg=[90.0, 180.0], duration=2e-6, rabi=1e4)
    np.testing.assert_allclose(s.phase, [np.pi / 2, np.pi])


def test_waveform_negative_amplitude_flips_phase():
    s = waveform([-0.5, 0.5], [0.25, 0.25], duration=2e-6, rabi=1e4)
    np.testing.assert_allclose(s.amp, [0.5, 0.5])
    np.testing.assert_allclose(s.phase, [0.25 + np.pi, 0.25])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"amp": [2.0], "phase": [0.0]}, "normalised"),
        ({"amp": [1.0], "phase": [0.0], "i": [1.0], "q": [0.0]}, "either"),
        ({}, "either"),
        ({"i": [1.0]}, "i and q"),
        ({"amp": [1.0]}, "phase"),
        ({"amp": [1.0], "phase": [0.0], "phase_deg": [0.0]}, "phase"),
        ({"amp": [1.0, 1.0], "phase": [0.0]}, "shape"),
        ({"i": [1.0, 1.0], "q": [0.0]}, "shape"),
        ({"amp": [1.0], "phase": [0.0], "rabi": None}, "rabi"),
        ({"amp": [1.0], "phase": [0.0], "t90": 1e-6}, "rabi"),
        ({"amp": [1.0], "phase": [0.0], "rabi": -1.0}, "rabi"),
        ({"amp": [1.0], "phase": [0.0], "rabi": None, "t90": 0.0}, "t90"),
    ],
)
def test_waveform_validation(kwargs, match):
    kwargs = {"rabi": 1e4, **kwargs}
    with pytest.raises(ValueError, match=match):
        waveform(**kwargs, duration=1e-6)
