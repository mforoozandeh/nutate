import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from helpers import hard

from nutate import PhaseCycle, Style, run, waveform

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
    with pytest.raises(ValueError, match="video, still or b1"):
        run(_pulse(), 100e3, video=False, out=tmp_path)


def test_run_explicit_offsets(tmp_path):
    result = run(
        _pulse(), offsets=[-1e3, 0.0, 2e3], video=False, b1=True, out=tmp_path, b1_scales=[0.9, 1.1]
    )
    np.testing.assert_array_equal(result.b1_map.offsets, [-1e3, 0.0, 2e3])


@pytest.mark.parametrize("kwargs", [{}, {"bandwidth": 1e3, "offsets": [0.0, 1.0]}])
def test_run_needs_bandwidth_or_offsets(kwargs, tmp_path):
    with pytest.raises(ValueError, match="bandwidth or offsets"):
        run(_pulse(), video=False, b1=True, out=tmp_path, **kwargs)


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


@pytest.mark.render
def test_run_still_uses_scene_options(tmp_path):
    result = run(
        _pulse(),
        offsets=[-1e3, 0.0, 2e3],
        video=False,
        still=True,
        out=tmp_path,
        name="p",
        video_options={"duration": 0.1, "hold": 0, "size": (320, 240), "style": Style("dark")},
    )
    assert result.files == [tmp_path / "p.png"]
    img = iio.imread(result.files[0])
    assert img.shape[:2] == (240, 320)
    np.testing.assert_allclose(img[2, 2, :3], (15, 17, 21), atol=3)  # dark background
    np.testing.assert_array_equal(result.trajectory.offsets, [-1e3, 0.0, 2e3])


@pytest.mark.render
def test_run_show_opens_windows_and_writes_nothing(tmp_path, monkeypatch):
    seen, before = [], set(plt.get_fignums())
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: seen.append("3d"))
    monkeypatch.setattr(plt, "show", lambda *a, **k: seen.append(set(plt.get_fignums()) - before))
    result = run(
        _pulse(),
        100e3,
        b1=True,
        show=True,
        out=tmp_path / "res",
        n_offsets=3,
        b1_offsets=3,
        b1_scales=[0.9, 1.1],
        video_options={"duration": 0.1},
    )
    assert seen[0] == "3d" and len(seen[1]) == 3  # viewer, then the three B1 figures
    assert result.files == []
    assert not (tmp_path / "res").exists()
    assert result.trajectory is not None and result.b1_map is not None
    plt.close("all")
