import numpy as np
import pytest

from nutate import PhaseCycle, hard, run, waveform

T90 = 10e-6


def _pulse():
    return waveform(np.ones(20), np.zeros(20), duration=T90, t90=T90)


def test_run_b1_writes_figures_and_map(tmp_path):
    result = run(
        _pulse(),
        100e3,
        video=False,
        b1=True,
        out=tmp_path / "res",
        name="p",
        b1_offsets=5,
        b1_scales=[0.5, 1.0, 1.5],
    )
    assert result.files == [tmp_path / "res" / f"p_b1_{s}.png" for s in ("map", "surface", "phase")]
    assert all(f.exists() for f in result.files)
    assert result.trajectory is None
    np.testing.assert_allclose(result.b1_map.offsets, [-50e3, -25e3, 0, 25e3, 50e3])
    np.testing.assert_allclose(result.b1_map.scales, [0.5, 1.0, 1.5])
    flips = np.deg2rad([45, 90, 135])
    np.testing.assert_allclose(result.b1_map.m[:, 2, 1], -np.sin(flips), atol=1e-9)
    assert result.b1_phase.shape == (3,)


def test_run_b1_default_scales(tmp_path):
    result = run(_pulse(), 100e3, video=False, b1=True, out=tmp_path, b1_offsets=3)
    np.testing.assert_allclose(result.b1_map.scales, np.linspace(0.5, 1.5, 101))


def test_run_accepts_list_and_phase_cycle(tmp_path):
    result = run(
        [hard(T90 / 2, flip_deg=45), hard(T90 / 2, flip_deg=45)],
        100e3,
        video=False,
        b1=True,
        out=tmp_path,
        b1_offsets=3,
        b1_scales=[0.5, 1.0],
        phase_cycle=PhaseCycle([[90.0, 90.0]], [0.0]),
    )
    np.testing.assert_allclose(result.b1_map.m[1, 1], [1, 0, 0], atol=1e-9)


@pytest.mark.parametrize("grid", [{"b1_offsets": 1}, {"b1_scales": [1.0]}])
def test_run_b1_grid_needs_two_points_per_axis(grid, tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        run(_pulse(), 100e3, video=False, b1=True, out=tmp_path, **{"b1_offsets": 3, **grid})


def test_run_needs_an_output(tmp_path):
    with pytest.raises(ValueError, match="video or b1"):
        run(_pulse(), 100e3, video=False, out=tmp_path)


@pytest.mark.parametrize("bandwidth", [0.0, -1e3, np.inf])
def test_run_bandwidth_validation(bandwidth, tmp_path):
    with pytest.raises(ValueError, match="bandwidth"):
        run(_pulse(), bandwidth, b1=True, video=False, out=tmp_path)


@pytest.mark.render
def test_run_video(tmp_path):
    result = run(
        _pulse(),
        100e3,
        out=tmp_path,
        name="p",
        n_offsets=5,
        video_options={"duration": 0.1, "hold": 0, "size": (320, 240)},
    )
    assert result.files == [tmp_path / "p.mp4"]
    assert result.files[0].stat().st_size > 0
    np.testing.assert_allclose(result.trajectory.offsets, [-50e3, -25e3, 0, 25e3, 50e3])
    np.testing.assert_allclose(result.trajectory.final[2], [0, -1, 0], atol=1e-9)
    assert result.b1_map is None
