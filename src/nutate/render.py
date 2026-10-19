"""Render a :class:`~nutate.simulate.Trajectory` as a Bloch-sphere video or still."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import PIL.Image
import pyvista as pv

from nutate._scene import THEMES, Camera, Style, _camera_position, _Scene, _trail_window
from nutate.simulate import Trajectory

_VIDEO_DEFAULTS = {
    ".mp4": {"fps": 30, "size": (1920, 1080)},
    ".gif": {"fps": 15, "size": (960, 540)},
}


def _frame_steps(n_steps: int, n_frames: int) -> np.ndarray:
    """Simulation step shown in each of ``n_frames`` frames, evenly spread over 0…n_steps."""
    if n_frames <= 1:
        return np.array([n_steps])
    if n_frames - 1 > n_steps:
        warnings.warn(
            f"{n_frames} frames but only {n_steps} simulation steps: frames repeat; "
            "simulate with more steps for smooth motion",
            UserWarning,
            stacklevel=3,
        )
    return np.round(np.arange(n_frames) * n_steps / (n_frames - 1)).astype(int)


def _offset_indices(n_off: int, offsets: None | int | Sequence[int]) -> np.ndarray:
    """Indices of the offsets to draw: all, ``k`` evenly spaced, or explicit indices."""
    if offsets is None:
        return np.arange(n_off)
    if isinstance(offsets, (int, np.integer)) and not isinstance(offsets, bool):
        if not 1 <= offsets <= n_off:
            raise ValueError(f"offsets={offsets}: need between 1 and {n_off} offsets")
        return np.round(np.linspace(0, n_off - 1, offsets)).astype(int)
    idx = np.asarray(offsets, dtype=int)
    if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= n_off:
        raise ValueError(f"offsets must be indices in [0, {n_off}), got {list(offsets)}")
    return idx


def _check_suffix(path: Path, allowed: Sequence[str]) -> None:
    if path.suffix.lower() not in allowed:
        raise ValueError(
            f"unsupported output {path.suffix or path.name!r}; use one of {', '.join(allowed)}"
        )


class _GifWriter:
    """Collects quantized frames; :meth:`close` writes a looping GIF (Pillow merges repeats)."""

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.duration = round(1000 / fps)
        self.frames: list[PIL.Image.Image] = []

    def append_data(self, image: np.ndarray) -> None:
        self.frames.append(PIL.Image.fromarray(image).quantize(256))

    def close(self) -> None:
        if self.frames:
            self.frames[0].save(
                self.path,
                save_all=True,
                append_images=self.frames[1:],
                duration=self.duration,
                loop=0,
                optimize=True,
            )


def _writer(path: Path, fps: int):
    if path.suffix.lower() == ".gif":
        return _GifWriter(path, fps)
    return imageio.v2.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=8,  # 1080 px is not a multiple of the default 16
    )


def _make_scene(traj, size, offsets, **scene_kwargs) -> _Scene:
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        return _Scene(
            plotter, traj, offsets=_offset_indices(traj.offsets.size, offsets), **scene_kwargs
        )
    except Exception:
        plotter.close()
        raise


def render(
    traj: Trajectory,
    path: str | os.PathLike,
    *,
    duration: float = 10.0,
    fps: int | None = None,
    hold: float = 1.0,
    size: tuple[int, int] | None = None,
    offsets: None | int | Sequence[int] = None,
    mode: Literal["arrows", "ribbon", "both"] = "arrows",
    trail: float = 0.1,
    inset: bool = True,
    inset_phase: bool = False,
    theme: Literal["light", "dark"] = "light",
    style: Style = Style(),
    camera: Camera = Camera(),
) -> Path:
    """Write a video of ``traj`` to ``path`` (``.mp4`` or ``.gif``).

    ``duration`` s of motion at ``fps``, then the last frame held for ``hold`` s.
    ``size`` is (width, height) in pixels. ``offsets``: ``None`` = all, ``int k`` = k evenly
    spaced, or a sequence of indices. ``mode`` draws arrows, a ribbon through the tips, or
    both; ``trail`` is the fraction of the sequence shown as fading tip trails (0 = off).
    ``inset`` adds the RF amplitude vs time with a cursor; ``inset_phase`` adds the phase.
    ``theme`` is ``"light"`` or ``"dark"``; ``style`` fine-tunes colours, sizes and font.
    Parent directories are created.
    """
    path = Path(path)
    _check_suffix(path, list(_VIDEO_DEFAULTS))
    defaults = _VIDEO_DEFAULTS[path.suffix.lower()]
    fps = fps or defaults["fps"]
    size = size or defaults["size"]
    n_frames = max(1, round(duration * fps))
    steps = _frame_steps(traj.t.size - 1, n_frames)
    path.parent.mkdir(parents=True, exist_ok=True)

    scene = _make_scene(
        traj,
        size,
        offsets,
        mode=mode,
        trail=trail,
        inset=inset,
        inset_phase=inset_phase,
        theme=theme,
        style=style,
        camera=camera,
    )
    writer = _writer(path, fps)
    try:
        for i, step in enumerate(steps):
            scene.update(int(step), i / max(n_frames - 1, 1))
            image = scene.screenshot()
            writer.append_data(image)
        for _ in range(round(hold * fps)):
            writer.append_data(image)
    finally:
        writer.close()
        scene.plotter.close()
    return path


def snapshot(
    traj: Trajectory,
    path: str | os.PathLike,
    *,
    step: int = -1,
    size: tuple[int, int] = (1920, 1080),
    offsets: None | int | Sequence[int] = None,
    mode: Literal["arrows", "ribbon", "both"] = "arrows",
    trail: float = 0.1,
    inset: bool = True,
    inset_phase: bool = False,
    theme: Literal["light", "dark"] = "light",
    style: Style = Style(),
    camera: Camera = Camera(),
) -> Path:
    """Write one frame of ``traj`` at simulation ``step`` (``-1`` = last) to a ``.png``.

    Other options as :func:`render`."""
    path = Path(path)
    _check_suffix(path, [".png"])
    path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = traj.t.size - 1
    if not -(n_steps + 1) <= step <= n_steps:
        raise ValueError(f"step must be in [{-(n_steps + 1)}, {n_steps}], got {step}")
    scene = _make_scene(
        traj,
        size,
        offsets,
        mode=mode,
        trail=trail,
        inset=inset,
        inset_phase=inset_phase,
        theme=theme,
        style=style,
        camera=camera,
    )
    try:
        scene.update(step % (n_steps + 1))
        imageio.v3.imwrite(path, scene.screenshot())
    finally:
        scene.plotter.close()
    return path


def show(
    traj: Trajectory,
    *,
    offsets: None | int | Sequence[int] = None,
    mode: Literal["arrows", "ribbon", "both"] = "arrows",
    trail: float = 0.1,
    inset: bool = True,
    inset_phase: bool = False,
    theme: Literal["light", "dark"] = "light",
    style: Style = Style(),
    camera: Camera = Camera(),
) -> None:
    """Open an interactive window with a slider to scrub through ``traj``.

    Options as :func:`render`. Needs a display.
    """
    plotter = pv.Plotter()
    scene = _Scene(
        plotter,
        traj,
        offsets=_offset_indices(traj.offsets.size, offsets),
        mode=mode,
        trail=trail,
        inset=inset,
        inset_phase=inset_phase,
        theme=theme,
        style=style,
        camera=camera,
    )
    n_steps = traj.t.size - 1
    plotter.add_slider_widget(
        lambda value: scene.update(round(value)),
        rng=(0, n_steps),
        value=n_steps,
        title="step",
        pointa=(0.05, 0.08),
        pointb=(0.45, 0.08),
        color=THEMES[theme]["text"],
        fmt="%.0f",
        interaction_event="always",  # follow the drag, not just the release
    )
    scene.update(n_steps)
    plotter.show()


__all__ = [
    "THEMES",
    "Camera",
    "Style",
    "_camera_position",
    "_trail_window",
    "render",
    "show",
    "snapshot",
]
