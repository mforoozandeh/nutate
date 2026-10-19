import numpy as np
import pytest
from nutate.pulses.swept_frequency import (
    CHORUS_CYCLE,
    chorus,
    chorus_phase_correction,
    phase_correction,
    q_factor,
)

from nutate import simulate


def test_chorus_structure():
    segs = chorus(2e-3, 1e-3, 300e3, 1e-6)
    assert [s.name for s in segs] == ["90° chirp", "180° chirp", "delay", "180° chirp"]
    assert [s.n_steps for s in segs] == [2000, 2000, 1000, 1000]
    np.testing.assert_allclose([s.duration for s in segs], [2e-3, 2e-3, 1e-3, 1e-3])
    assert segs[2].dt == pytest.approx(1e-6)
    assert segs[0].rabi == pytest.approx(np.sqrt(2e-3 * 300e3 * q_factor(90) / (2 * np.pi)) / 2e-3)
    assert segs[1].rabi == pytest.approx(np.sqrt(2e-3 * 300e3 * 5 / (2 * np.pi)) / 2e-3)
    assert [s.amp.max() for s in segs] == pytest.approx([1, 1, 0, 1])


@pytest.mark.parametrize(
    "args, steps",
    [
        ((10e-3, 4e-3, 20e3, 2e-5), [500, 450, 250, 200]),  # 10e-3/2e-5 floors to 499 naively
        ((200e-9, 100e-9, 200e6, 0.625e-9), [320, 320, 160, 160]),  # EPR, Bruker AWG step
    ],
)
def test_chorus_step_counts_robust_to_float_division(args, steps):
    assert [s.n_steps for s in chorus(*args)] == steps


def test_chorus_cycle_table():
    p, r = CHORUS_CYCLE.pulse_phases_deg, CHORUS_CYCLE.receiver_deg
    assert p.shape == (16, 4)
    assert not p[:, 0].any()
    assert not p[:, 2].any()
    np.testing.assert_array_equal(p[:4, 1], [0, 90, 180, 270])
    np.testing.assert_array_equal(p[4:8, 3], 90)
    np.testing.assert_array_equal(r[:8], [0, 180, 0, 180, 180, 0, 180, 0])


BAND = np.linspace(-120e3, 120e3, 41)  # central 80 % of 300 kHz


def test_chorus_excites_band():
    final = simulate(chorus(2e-3, 1e-3, 300e3, 1e-6), BAND, phase_cycle=CHORUS_CYCLE).final
    assert np.hypot(final[:, 0], final[:, 1]).mean() > 0.9


def _phase_spread(sequence):
    f = simulate(sequence, BAND, phase_cycle=CHORUS_CYCLE).final
    return np.unwrap(np.angle(f[:, 1] + 1j * f[:, 0])).std()


def test_phase_correction_reduces_dispersion():
    corr = chorus_phase_correction(2e-3, 1e-3, 300e3, 1e-6, degree=10)
    assert corr.shape == (2000,)
    corrected = _phase_spread(chorus(2e-3, 1e-3, 300e3, 1e-6, phase_correction_deg=corr))
    assert corrected < _phase_spread(chorus(2e-3, 1e-3, 300e3, 1e-6))


def test_phase_correction_uses_q180():
    default = chorus_phase_correction(2e-3, 1e-3, 300e3, 2e-6, degree=10)
    gentler = chorus_phase_correction(2e-3, 1e-3, 300e3, 2e-6, degree=10, q180=3.0)
    assert np.abs(default - gentler).max() > 1e-3


def test_generic_phase_correction_matches_chorus_version():
    sequence = chorus(2e-3, 1e-3, 300e3, 2e-6, q180=4.0)
    generic = phase_correction(sequence, 300e3, phase_cycle=CHORUS_CYCLE, degree=10)
    specific = chorus_phase_correction(2e-3, 1e-3, 300e3, 2e-6, q180=4.0, degree=10)
    np.testing.assert_allclose(generic, specific, atol=1e-12)
    assert generic.shape == (sequence[0].n_steps,)
