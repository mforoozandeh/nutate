import warnings

import imageio
import imageio.v3 as iio
import numpy as np
import PIL.Image
import pytest
import pyvista as pv
from helpers import hard

from nutate import Camera, Style, Trajectory, render, show, simulate, snapshot
from nutate.render import _camera_position, _frame_steps, _offset_indices, _trail_window


def test_frame_steps():
    s = _frame_steps(500, 301)
    assert len(s) == 301
    assert s[0] == 0
    assert s[-1] == 500
    assert (np.diff(s) >= 0).all()
    assert list(_frame_steps(7, 1)) == [7]


def test_frame_steps_warns_when_short():  # Review Focus 4
    with pytest.warns(UserWarning, match="more steps"):
        s = _frame_steps(10, 300)
    assert s.min() == 0
    assert s.max() == 10


def test_offset_indices():
    assert list(_offset_indices(31, None)) == list(range(31))
    k = _offset_indices(31, 5)
    assert len(set(k)) == 5
    assert k[0] == 0
    assert k[-1] == 30
    assert list(_offset_indices(31, [0, 3])) == [0, 3]
    for bad in (40, [31], 0):
        with pytest.raises(ValueError, match="offsets"):
            _offset_indices(31, bad)


def test_camera_azimuth_elevation_convention():
    np.testing.assert_allclose(_camera_position(Camera(0, 0), 1.0), (0, -1, 0), atol=1e-12)
    np.testing.assert_allclose(_camera_position(Camera(90, 0), 1.0), (1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(
        _camera_position(Camera(), 1.0), (0.65328, -0.65328, 0.38268), atol=1e-5
    )
    np.testing.assert_allclose(
        _camera_position(Camera(0, 0, orbit=90), 1.0, fraction=1.0), (1, 0, 0), atol=1e-12
    )


@pytest.fixture
def traj():
    return simulate([hard(10e-6, n_steps=100)], np.linspace(-5e4, 5e4, 5))


SMALL = (320, 240)


@pytest.mark.render
def test_snapshot_not_blank(traj, tmp_path):
    img = iio.imread(snapshot(traj, tmp_path / "a.png", size=SMALL))
    assert img.shape[:2] == (240, 320)
    assert img.std() > 5


@pytest.mark.render
@pytest.mark.parametrize("theme, corner", [("light", (255, 255, 255)), ("dark", (15, 17, 21))])
def test_theme_background(traj, tmp_path, theme, corner):
    img = iio.imread(snapshot(traj, tmp_path / "a.png", size=SMALL, theme=theme))
    np.testing.assert_allclose(img[2, 2, :3], corner, atol=3)


@pytest.mark.render
def test_unknown_theme(traj, tmp_path):
    with pytest.raises(ValueError, match="theme"):
        snapshot(traj, tmp_path / "a.png", size=SMALL, theme="blue")


@pytest.mark.render
def test_render_mp4_frame_count(traj, tmp_path):
    p = render(traj, tmp_path / "a.mp4", duration=10 / 30, fps=30, hold=5 / 30, size=SMALL)
    assert imageio.v2.get_reader(p, format="FFMPEG").count_frames() == 15


@pytest.mark.render
def test_render_creates_parent_dirs(traj, tmp_path):  # Review Focus 5
    out = render(traj, tmp_path / "new" / "sub" / "a.mp4", duration=0.1, hold=0, size=SMALL)
    assert out.exists()


@pytest.mark.render
def test_single_zero_offset(tmp_path):  # Review Focus 2
    t = simulate([hard(10e-6, n_steps=20)], [0.0])
    assert iio.imread(snapshot(t, tmp_path / "a.png", size=SMALL)).std() > 5


@pytest.mark.render
def test_zero_length_vector(traj, tmp_path):  # Review Focus 3
    m = traj.m.copy()
    m[:, 2] = 0.0
    t = Trajectory(traj.t, traj.offsets, m, traj.segments, traj.sequence)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        snapshot(t, tmp_path / "a.png", size=SMALL)


@pytest.mark.render
def test_single_colour_style(traj, tmp_path):  # one colour for every arrow
    a = iio.imread(snapshot(traj, tmp_path / "a.png", size=SMALL)).astype(float)
    b = iio.imread(snapshot(traj, tmp_path / "b.png", size=SMALL, style=Style(color="gray")))
    assert np.abs(a - b).mean() > 0.5


@pytest.mark.render
def test_bad_extensions(traj, tmp_path):
    with pytest.raises(ValueError, match=r"\.mp4"):
        render(traj, tmp_path / "a.avi")
    with pytest.raises(ValueError, match=r"\.png"):
        snapshot(traj, tmp_path / "a.mp4")


def test_trail_window():
    w = _trail_window(100, 500, 0.1)
    assert w[0] == 50
    assert w[-1] == 100
    assert len(w) <= 200
    assert len(_trail_window(100, 500, 0.0)) == 0
    w = _trail_window(500, 500, 1.0)
    assert w[0] == 0
    assert w[-1] == 500
    assert len(w) == 200


@pytest.mark.render
@pytest.mark.parametrize("mode", ["ribbon", "both"])
def test_modes_render(traj, tmp_path, mode):
    img = iio.imread(snapshot(traj, tmp_path / "a.png", size=SMALL, mode=mode, trail=1.0))
    assert img.std() > 5


@pytest.mark.render
def test_trail_changes_image(traj, tmp_path):
    a = iio.imread(snapshot(traj, tmp_path / "a.png", size=SMALL, trail=0.0)).astype(float)
    b = iio.imread(snapshot(traj, tmp_path / "b.png", size=SMALL, trail=1.0)).astype(float)
    assert np.abs(a - b).mean() > 0.5


@pytest.mark.render
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mode": "ribbon", "offsets": [2]}, "ribbon"),
        ({"mode": "dots"}, "mode"),
        ({"trail": 1.5}, "trail"),
        ({"trail": -0.1}, "trail"),
    ],
)
def test_mode_trail_validation(traj, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        snapshot(traj, tmp_path / "a.png", size=SMALL, **kwargs)


def _region(img):  # bottom-right inset area
    h, w = img.shape[:2]
    return img[int(0.55 * h) :, int(0.55 * w) :].astype(float)


@pytest.mark.render
def test_inset_visible(traj, tmp_path):
    a = iio.imread(snapshot(traj, tmp_path / "a.png", size=(640, 360), inset=False))
    b = iio.imread(snapshot(traj, tmp_path / "b.png", size=(640, 360), inset=True))
    assert np.abs(_region(a) - _region(b)).mean() > 1.0


@pytest.mark.render
def test_inset_phase_panel_visible(traj, tmp_path):
    a = iio.imread(snapshot(traj, tmp_path / "a.png", size=(640, 360), inset_phase=False))
    b = iio.imread(snapshot(traj, tmp_path / "b.png", size=(640, 360), inset_phase=True))
    assert np.abs(_region(a) - _region(b)).mean() > 0.5


@pytest.mark.render
def test_render_gif(traj, tmp_path):
    p = render(traj, tmp_path / "a.gif", duration=5 / 15, hold=0, size=SMALL)
    with PIL.Image.open(p) as im:
        assert im.n_frames == 5
        assert im.info["loop"] == 0
        assert abs(im.info["duration"] - 1000 / 15) <= 10  # GIF stores 1/100 s


@pytest.mark.render
def test_show_builds_slider(traj, monkeypatch):
    seen = {}
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: seen.setdefault("p", self))
    show(traj)
    widgets = getattr(seen["p"], "widgets", seen["p"])  # PyVista ≥ 0.46 moved them
    assert len(widgets.slider_widgets) == 1


def _overlay_texts(plotter):
    actors = plotter.renderer.GetViewProps()
    actors.InitTraversal()
    texts = []
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextProp()
        if hasattr(actor, "GetInput"):
            texts.append(actor.GetInput())
    return texts


@pytest.mark.render
def test_show_slider_keeps_user_camera(traj, monkeypatch):
    seen = {}
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: seen.setdefault("p", self))
    show(traj)
    p = seen["p"]
    try:
        slider = getattr(p, "widgets", p).slider_widgets[0]
        p.camera.position = (1.0, 6.0, 2.5)
        p.camera.view_angle = 12.5
        slider.GetRepresentation().SetValue(10)
        slider.InvokeEvent("InteractionEvent")  # mid-drag, not only on release
        assert any(t.startswith("t = 1.00 µs") for t in _overlay_texts(p))
        np.testing.assert_allclose(p.camera.position, (1.0, 6.0, 2.5))
        assert p.camera.view_angle == 12.5
    finally:
        p.close()


@pytest.mark.render
@pytest.mark.parametrize("step", [101, -102])
def test_snapshot_step_out_of_range(traj, tmp_path, step):  # traj has 100 steps (states 0..100)
    with pytest.raises(ValueError, match="step"):
        snapshot(traj, tmp_path / "a.png", step=step, size=SMALL)


@pytest.mark.render
def test_inset_plots_rf_in_hz(traj):  # amp · rabi, not the normalised waveform
    from nutate.render import _make_scene

    scene = _make_scene(
        traj,
        SMALL,
        None,
        mode="arrows",
        trail=0.1,
        inset=True,
        inset_phase=False,
        style=Style(),
        camera=Camera(),
    )
    try:
        (chart,) = scene._overlay._charts
        assert chart.y_label == "B1 (kHz)"
        assert chart.y_range == pytest.approx([0.0, 1.1 * 25.0])  # hard 10 µs 90°: 25 kHz
    finally:
        scene.plotter.close()
