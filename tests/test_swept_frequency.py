import numpy as np
import pytest
from nutate.pulses.swept_frequency import chirp, hs, q_factor

from nutate import simulate


def test_q_factor_90():
    assert q_factor(90) == pytest.approx(-2 * np.log(0.5) / np.pi, rel=1e-12)  # ≈ 0.441271


@pytest.mark.parametrize("flip", [0, 180, 200])
def test_q_factor_invalid(flip):
    with pytest.raises(ValueError, match="q=5"):
        q_factor(flip)


def test_chirp_envelope():
    c = chirp(10e-3, 20e3, 500, smoothing=10)
    peak = np.sqrt(10e-3 * 20e3 * q_factor(90) / (2 * np.pi)) / 10e-3
    assert c.n_steps == 500
    assert c.dt == pytest.approx(2e-5)
    assert c.rabi == pytest.approx(peak)
    ramp = np.sin(np.linspace(0, np.pi / 2, 50))
    np.testing.assert_allclose(c.amp[:50], ramp)
    np.testing.assert_allclose(c.amp[50:450], 1.0)
    np.testing.assert_allclose(c.amp[450:], ramp[::-1])


def test_chirp_phase_quadratic():
    c = chirp(10e-3, 20e3, 500)
    t = np.linspace(0, 10e-3, 500)
    expected = 2 * np.pi * (20e3 / t[-1] * t**2 / 2 - 10e3 * t)
    np.testing.assert_allclose(c.phase, expected - expected.min(), atol=1e-9)


def test_chirp_phase_correction_added():
    corr = np.linspace(0, 90, 500)
    np.testing.assert_allclose(
        chirp(10e-3, 20e3, 500, phase_correction_deg=corr).phase - chirp(10e-3, 20e3, 500).phase,
        np.deg2rad(corr),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"smoothing": 50}, {"smoothing": -1}, {"phase_correction_deg": np.zeros(3)}],
)
def test_chirp_validation(kwargs):
    with pytest.raises(ValueError):
        chirp(10e-3, 20e3, 500, **kwargs)


def test_chirp_180_inverts_band():
    c = chirp(2e-3, 100e3, 2000, smoothing=10, q=5)
    final = simulate([c], np.linspace(-40e3, 40e3, 41)).final
    assert (final[:, 2] < -0.95).all()


def test_hs_shape():
    # sech amplitude, phase phi0·ln cosh, βT = 10.6.
    T, bw, n = 2e-3, 100e3, 401
    s = hs(T, bw, n, rabi=5e3, phase_deg=90)
    t = np.linspace(-T / 2, T / 2, n)
    np.testing.assert_allclose(s.amp, 1 / np.cosh(10.6 * t / T))
    assert s.rabi == 5e3
    phi0 = np.pi * bw * T / 10.6
    np.testing.assert_allclose(s.phase, np.pi / 2 + phi0 * np.log(np.cosh(10.6 * t / T)))
    assert s.dt == pytest.approx(T / n)
    assert s.n_steps == n


def test_hs_sweeps_bandwidth():
    T, bw = 2e-3, 100e3
    s = hs(T, bw, 2001, rabi=5e3)
    freq = np.gradient(s.phase, np.linspace(-T / 2, T / 2, 2001)) / (2 * np.pi)
    assert freq[-1] == pytest.approx(bw / 2 * np.tanh(5.3), rel=1e-3)
    assert freq[0] == pytest.approx(-bw / 2 * np.tanh(5.3), rel=1e-3)
