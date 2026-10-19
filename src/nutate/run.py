"""One call from a pulse to its Bloch-sphere video and B1 robustness figures."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from nutate.b1 import B1Map, b1_map, b1_phase_sensitivity
from nutate.profiles import plot_b1_map, plot_b1_phase
from nutate.render import render, snapshot
from nutate.render import show as show_scene
from nutate.segment import Segment
from nutate.simulate import PhaseCycle, Trajectory, simulate

# video_options that also apply to the still and the interactive viewer.
_SCENE_OPTIONS = ("offsets", "mode", "trail", "inset", "inset_phase", "theme", "style", "camera")


@dataclass(frozen=True, eq=False)
class RunResult:
    """What :func:`run` simulated and wrote.

    Attributes
    ----------
    trajectory : the video's trajectory, or ``None`` without ``video`` or ``still``.
    b1_map : final magnetization over offsets × B1 scales, or ``None`` without ``b1``.
    b1_phase : on-resonance phase change (degrees) per B1 scale, or ``None`` without ``b1``.
    files : paths written.
    """

    trajectory: Trajectory | None
    b1_map: B1Map | None
    b1_phase: np.ndarray | None
    files: list[Path]


def run(
    sequence: Segment | Sequence[Segment],
    bandwidth: float | None = None,
    *,
    offsets: ArrayLike | None = None,
    video: bool = True,
    still: bool = False,
    b1: bool = False,
    show: bool = False,
    out: str | os.PathLike = ".",
    name: str = "pulse",
    n_offsets: int = 31,
    b1_offsets: int = 101,
    b1_scales: ArrayLike | None = None,
    component: Literal["x", "y", "z"] = "y",
    sign: float = -1,
    rf_ref: float | None = None,
    phase_cycle: PhaseCycle | None = None,
    video_options: Mapping[str, Any] | None = None,
) -> RunResult:
    """Simulate ``sequence`` (one pulse or a list, played back to back) and write its figures.

    Offsets span ``±bandwidth/2`` Hz: ``n_offsets`` for the video, ``b1_offsets`` for the B1
    map. Or give ``offsets`` (Hz) to use for both.

    - ``video`` writes ``<out>/<name>.mp4``; ``video_options`` go to
      :func:`~nutate.render.render`.
    - ``still`` writes ``<name>.png``, the last frame, with the ``video_options`` that apply.
    - ``b1`` writes ``<name>_b1_map.png`` and ``_b1_surface.png`` (``sign · M<component>``
      over the offsets and ``b1_scales``, default 101 scales from 0.5 to 1.5) and
      ``_b1_phase.png``. ``rf_ref`` (Hz) normalises the offset axis; it defaults to the
      highest Rabi frequency in ``sequence``.
    - ``show`` opens the interactive viewer and the B1 figures instead of writing files.
    """
    sequence = [sequence] if isinstance(sequence, Segment) else list(sequence)
    if (bandwidth is None) == (offsets is None):
        raise ValueError("give either bandwidth or offsets")
    if bandwidth is not None and not (np.isfinite(bandwidth) and bandwidth > 0):
        raise ValueError(f"bandwidth must be a positive frequency in Hz, got {bandwidth}")
    if not (video or still or b1):
        raise ValueError("nothing to do: set video, still or b1")

    def span(n: int) -> np.ndarray:
        if offsets is not None:
            return np.asarray(offsets, dtype=float)
        return np.linspace(-bandwidth / 2, bandwidth / 2, n)

    scales = np.linspace(0.5, 1.5, 101) if b1_scales is None else np.asarray(b1_scales)
    if b1 and (span(b1_offsets).size < 2 or scales.size < 2):
        raise ValueError("B1 maps need at least 2 offsets and at least 2 scales")
    options = dict(video_options or {})
    scene = {k: v for k, v in options.items() if k in _SCENE_OPTIONS}
    out = Path(out)
    trajectory = bmap = dphi = None
    files = []
    if video or still:
        trajectory = simulate(sequence, span(n_offsets), phase_cycle=phase_cycle)
        if show:
            show_scene(trajectory, **scene)
        else:
            if video:
                files.append(render(trajectory, out / f"{name}.mp4", **options))
            if still:
                size = {"size": options["size"]} if "size" in options else {}
                files.append(snapshot(trajectory, out / f"{name}.png", **scene, **size))
    if b1:
        bmap = b1_map(sequence, span(b1_offsets), scales, rf_ref=rf_ref, phase_cycle=phase_cycle)
        dphi = b1_phase_sensitivity(sequence, scales, phase_cycle=phase_cycle)
        figures = {
            "map": plot_b1_map(bmap, component, sign=sign),
            "surface": plot_b1_map(bmap, component, sign=sign, surface=True),
            "phase": plot_b1_phase(scales, dphi),
        }
        if show:
            plt.show()
        else:
            out.mkdir(parents=True, exist_ok=True)
            for suffix, fig in figures.items():
                files.append(out / f"{name}_b1_{suffix}.png")
                fig.savefig(files[-1], dpi=150)
                plt.close(fig)
    return RunResult(trajectory, bmap, dphi, files)
