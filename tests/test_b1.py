import numpy as np
import pytest
from helpers import hard

from nutate import PhaseCycle, b1_map, b1_phase_sensitivity, simulate_final, waveform
from nutate.pulses.composite import PRESETS, composite


def test_b1_map_hard_pulse_on_resonance():
    bmap = b1_map([hard(10e-6)], [0.0], [0.5, 1.0, 1.5])
    flips = np.deg2rad(90 * np.array([0.5, 1.0, 1.5]))
    assert bmap.m.shape == (3, 1, 3)
    np.testing.assert_allclose(bmap.m[:, 0, 1], -np.sin(flips), atol=1e-9)
    np.testing.assert_allclose(bmap.m[:, 0, 2], np.cos(flips), atol=1e-9)
    assert bmap.rabi_ref == pytest.approx(25e3)  # default: the highest Rabi frequency


def test_b1_map_matches_scaled_simulation():
    seq = composite(10e-6, *PRESETS["4"])
    offsets = np.linspace(-5e4, 5e4, 7)
    scales = np.array([0.8, 1.1])
    bmap = b1_map(seq, offsets, scales, rabi_ref=1e4)
    for i, z in enumerate(scales):
        np.testing.assert_allclose(bmap.m[i], simulate_final(seq, offsets, b1=z), atol=1e-12)
    np.testing.assert_array_equal(bmap.offsets, offsets)
    np.testing.assert_array_equal(bmap.scales, scales)
    assert bmap.rabi_ref == 1e4


def test_b1_map_uses_phase_cycle():
    pc = PhaseCycle([[90.0]], [0.0])
    offsets = [0.0, 1e4]
    np.testing.assert_allclose(
        b1_map([hard(1e-5)], offsets, [1.0], phase_cycle=pc).m[0],
        simulate_final([hard(1e-5)], offsets, phase_cycle=pc),
        atol=1e-12,
    )


def test_b1_phase_sensitivity_zero_for_hard_pulse():
    np.testing.assert_allclose(
        b1_phase_sensitivity([hard(10e-6)], np.linspace(0.5, 1.5, 5)), 0.0, atol=1e-9
    )


def test_b1_phase_sensitivity_definition():
    # Port of the "dphi/dB1 vs B1/B1_opt" sections: on resonance, phase at B1·(1+δ) minus
    # phase at B1·(1−δ), in degrees, wrapped to [−180, 180).
    seq = composite(10e-6, *PRESETS["3"])
    scales = np.array([0.7, 1.0, 1.3])
    d = b1_phase_sensitivity(seq, scales, delta=0.01)
    lo = simulate_final(seq, np.zeros(3), b1=scales * 0.99)
    hi = simulate_final(seq, np.zeros(3), b1=scales * 1.01)
    expected = np.rad2deg(np.angle(hi[:, 1] + 1j * hi[:, 0]) - np.angle(lo[:, 1] + 1j * lo[:, 0]))
    np.testing.assert_allclose(d, (expected + 180) % 360 - 180, atol=1e-9)
    assert np.all((d >= -180) & (d < 180))


@pytest.mark.parametrize("scales", [[], [0.5, -1.0], [[1.0]], [np.nan]])
def test_b1_scales_validation(scales):
    with pytest.raises(ValueError, match="scales"):
        b1_map([hard(1e-5)], [0.0], scales)


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1])
def test_b1_delta_validation(delta):
    with pytest.raises(ValueError, match="delta"):
        b1_phase_sensitivity([hard(1e-5)], [1.0], delta=delta)


def test_b1_map_default_rabi_ref_is_rabi():  # not the waveform's peak, which may sit below 1
    pulse = waveform([0.8, 0.9], [0.0, 0.0], duration=1e-6, rabi=1e4)
    assert b1_map([pulse], [-1e3, 1e3], [0.9, 1.1]).rabi_ref == 1e4
