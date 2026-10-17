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
from nutate.pulses import Segment
from nutate.render import render
from nutate.simulate import PhaseCycle, Trajectory, simulate


@dataclass(frozen=True, eq=False)
class RunResult:
    """What :func:`run` simulated and wrote.

    Attributes
    ----------
    trajectory : the video's trajectory, or ``None`` without ``video``.
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
    bandwidth: float,
    *,
    video: bool = True,
    b1: bool = False,
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

    Offsets span ``±bandwidth/2`` Hz. ``video`` writes ``<out>/<name>.mp4`` over ``n_offsets``
    offsets; ``video_options`` go to :func:`~nutate.render.render`. ``b1`` writes
    ``<name>_b1_map.png``, ``_b1_surface.png`` (``sign · M<component>`` over ``b1_offsets``
    offsets and ``b1_scales``, default 101 scales from 0.5 to 1.5) and ``_b1_phase.png``.
    ``rf_ref`` (Hz) normalises the B1 offset axis; it defaults to the sequence's peak RF,
    i.e. the Rabi frequency of a waveform normalised to 1.
    """
    sequence = [sequence] if isinstance(sequence, Segment) else list(sequence)
    if not (np.isfinite(bandwidth) and bandwidth > 0):
        raise ValueError(f"bandwidth must be a positive frequency in Hz, got {bandwidth}")
    if not (video or b1):
        raise ValueError("nothing to do: set video or b1")
    scales = np.linspace(0.5, 1.5, 101) if b1_scales is None else np.asarray(b1_scales)
    if b1 and (b1_offsets < 2 or scales.size < 2):
        raise ValueError("B1 maps need at least 2 offsets and at least 2 scales")
    out = Path(out)
    trajectory = bmap = dphi = None
    files = []
    if video:
        offsets = np.linspace(-bandwidth / 2, bandwidth / 2, n_offsets)
        trajectory = simulate(sequence, offsets, phase_cycle=phase_cycle)
        files.append(render(trajectory, out / f"{name}.mp4", **(video_options or {})))
    if b1:
        offsets = np.linspace(-bandwidth / 2, bandwidth / 2, b1_offsets)
        bmap = b1_map(sequence, offsets, scales, rf_ref=rf_ref, phase_cycle=phase_cycle)
        dphi = b1_phase_sensitivity(sequence, scales, phase_cycle=phase_cycle)
        figures = {
            "map": plot_b1_map(bmap, component, sign=sign),
            "surface": plot_b1_map(bmap, component, sign=sign, surface=True),
            "phase": plot_b1_phase(scales, dphi),
        }
        out.mkdir(parents=True, exist_ok=True)
        for suffix, fig in figures.items():
            files.append(out / f"{name}_b1_{suffix}.png")
            fig.savefig(files[-1], dpi=150)
            plt.close(fig)
    return RunResult(trajectory, bmap, dphi, files)
