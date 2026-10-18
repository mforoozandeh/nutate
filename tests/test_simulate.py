import numpy as np
import pytest

from nutate import PhaseCycle, Segment, delay, hard, simulate, simulate_final


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_matrix(omega, offs, phi, dt):  # literal matrix product of the convention
    w = np.sqrt((2 * np.pi * omega) ** 2 + (2 * np.pi * offs) ** 2)
    th = np.arctan2(omega, offs)
    return _rz(phi) @ _ry(th) @ _rz(dt * w) @ _ry(-th) @ _rz(-phi)


def _random_segment():
    rng = np.random.default_rng(0)
    amp = rng.uniform(0, 1, 50)
    amp[::7] = 0.0
    phase = rng.uniform(0, 2 * np.pi, 50)
    return Segment("r", amp, phase, 1e-5, 5e3)


OFFSETS = np.array([-3e3, -1e3, 0.0, 2e3, 7e3])


def test_matches_literal_rotation_matrices():
    seg = _random_segment()
    traj = simulate([seg], OFFSETS)
    for j, d in enumerate(OFFSETS):
        m = np.array([0.0, 0.0, 1.0])
        for k in range(seg.n_steps):
            m = _rotation_matrix(seg.rf[k], d, seg.phase[k], seg.dt) @ m
            np.testing.assert_allclose(traj.m[k + 1, j], m, atol=1e-12)


def test_norm_preserved():
    traj = simulate([_random_segment()], OFFSETS)
    np.testing.assert_allclose(np.linalg.norm(traj.m, axis=-1), 1.0, atol=1e-12)


def test_hard_90_conventions():
    np.testing.assert_allclose(simulate([hard(10e-6)], [0.0]).final[0], [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(
        simulate([hard(10e-6, phase_deg=90)], [0.0]).final[0], [1, 0, 0], atol=1e-9
    )


def test_delay_precession():
    traj = simulate([delay(1e-3, 100)], [250.0], m0=(1, 0, 0))
    np.testing.assert_allclose(traj.final[0], [0, 1, 0], atol=1e-12)  # 2π·250·1e-3 = π/2


def test_resonant_delay_is_identity():
    traj = simulate([delay(1e-3, 10)], [0.0])
    assert np.isfinite(traj.m).all()
    np.testing.assert_array_equal(traj.m, np.broadcast_to([0, 0, 1], traj.m.shape))


def test_trajectory_shape_and_segments():
    traj = simulate([hard(1e-5, n_steps=10), delay(1e-5, 5)], [-1e3, 0.0, 1e3])
    assert traj.t.shape == (16,)
    assert traj.m.shape == (16, 3, 3)
    assert traj.t[0] == 0
    assert traj.t[-1] == pytest.approx(2e-5)
    assert traj.segments == [("hard", 0, 10), ("delay", 10, 15)]
    np.testing.assert_array_equal(traj.final, traj.m[-1])


def test_offset_fn_remaps_offsets():
    a = simulate([delay(1e-3, 10, offset_fn=lambda d: 2 * d)], [125.0], m0=(1, 0, 0))
    b = simulate([delay(1e-3, 10)], [250.0], m0=(1, 0, 0))
    np.testing.assert_allclose(a.m, b.m, atol=1e-12)


def test_scalar_offset_is_one_offset():
    assert simulate([hard(1e-5, n_steps=4)], 0.0).m.shape == (5, 1, 3)


@pytest.mark.parametrize("bad", [[], [np.inf], [[0.0, 1.0]]])
def test_offsets_validation(bad):
    with pytest.raises(ValueError, match="offsets"):
        simulate([hard(1e-5)], bad)


def test_zero_cycle_equals_plain():
    seq = [hard(1e-5, n_steps=20), delay(1e-5, 10)]
    offs = [-2e4, 0.0, 3e4]
    a = simulate(seq, offs)
    b = simulate(seq, offs, phase_cycle=PhaseCycle(np.zeros((1, 2)), np.zeros(1)))
    np.testing.assert_allclose(a.m, b.m, atol=1e-12)


def test_receiver_phase_rotates_signal():
    pc = PhaseCycle([[0.0]], [90.0])
    np.testing.assert_allclose(
        simulate([hard(1e-5)], [0.0], phase_cycle=pc).final[0], [-1, 0, 0], atol=1e-9
    )  # Rz(-90°)·(0,-1,0)


def test_pulse_phase_added():
    pc = PhaseCycle([[90.0]], [0.0])
    np.testing.assert_allclose(
        simulate([hard(1e-5)], [0.0], phase_cycle=pc).final[0], [1, 0, 0], atol=1e-9
    )


def test_two_step_cycle_averages():
    pc = PhaseCycle([[0.0], [180.0]], [0.0, 180.0])
    np.testing.assert_allclose(
        simulate([hard(1e-5)], [0.0], phase_cycle=pc).final[0], [0, -1, 0], atol=1e-9
    )


def test_phase_cycle_shape_checks():
    with pytest.raises(ValueError, match="receiver"):
        PhaseCycle(np.zeros((2, 1)), np.zeros(3))
    with pytest.raises(ValueError, match="phase_cycle"):
        simulate(
            [hard(1e-5), delay(1e-5, 2)],
            [0.0],
            phase_cycle=PhaseCycle(np.zeros((2, 3)), np.zeros(2)),
        )


def test_b1_scales_rf_amplitude():
    final = simulate([hard(1e-5)], [0.0, 0.0], b1=[0.5, 1.0]).final
    np.testing.assert_allclose(final[0], [0, -np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-9)
    np.testing.assert_allclose(final[1], [0, -1, 0], atol=1e-9)


def test_scalar_b1_equals_scaled_segment():
    seg = _random_segment()
    scaled = seg.with_rabi(0.7 * seg.rabi)
    np.testing.assert_allclose(
        simulate([seg], OFFSETS, b1=0.7).m, simulate([scaled], OFFSETS).m, atol=1e-12
    )


def test_simulate_final_matches_trajectory():
    seq = [hard(1e-5, n_steps=20), delay(1e-5, 10)]
    pc = PhaseCycle([[0.0, 0.0], [90.0, 0.0]], [0.0, 90.0])
    b1 = np.array([0.8, 1.0, 1.2])
    offs = [-2e4, 0.0, 3e4]
    np.testing.assert_allclose(
        simulate_final(seq, offs, phase_cycle=pc, b1=b1),
        simulate(seq, offs, phase_cycle=pc, b1=b1).final,
        atol=1e-12,
    )


@pytest.mark.parametrize("bad", [-0.5, [1.0, 1.0], np.inf])
def test_b1_validation(bad):
    with pytest.raises(ValueError, match="b1"):
        simulate([hard(1e-5)], [0.0, 1e3, 2e3], b1=bad)
