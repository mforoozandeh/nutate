"""2D plots: excitation profiles vs offset, and one component over (time, offset)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from nutate._format import si_scale
from nutate.b1 import B1Map
from nutate.simulate import Trajectory

_LINE = "#2a78d6"  # single series: one hue; text and chrome stay neutral
_INK = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_COMPONENTS = {"x": 0, "y": 1, "z": 2}


def _style(ax: Axes) -> None:
    ax.grid(axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK, labelcolor=_INK)
    ax.yaxis.label.set_color(_INK)
    ax.xaxis.label.set_color(_INK)


def plot_profiles(traj: Trajectory, *, axes: Sequence[Axes] | None = None) -> Figure:
    """Final Ix, Iy, Iz, |Ixy| and phase/π vs offset, one panel each.

    Phase is ``angle(My + i·Mx)/π``.
    Pass five ``axes`` to draw into an existing figure.
    """
    if axes is None:
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(6, 8), layout="constrained")
    elif len(axes) != 5:
        raise ValueError(f"axes must hold 5 Axes, got {len(axes)}")
    else:
        fig = axes[0].figure
    f = traj.final
    factor, unit = si_scale(np.abs(traj.offsets).max(), "Hz")
    x = traj.offsets * factor
    panels = [
        ("Ix", f[:, 0]),
        ("Iy", f[:, 1]),
        ("Iz", f[:, 2]),
        ("|Ixy|", np.hypot(f[:, 0], f[:, 1])),
        ("phase/π", np.angle(f[:, 1] + 1j * f[:, 0]) / np.pi),
    ]
    for ax, (label, y) in zip(axes, panels):
        ax.plot(x, y, color=_LINE, linewidth=1.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(label)
        _style(ax)
    axes[-1].set_xlabel(f"offset ({unit})")
    return fig


def plot_map(
    traj: Trajectory,
    component: Literal["x", "y", "z"] = "y",
    *,
    sign: float = 1,
    ax: Axes | None = None,
) -> Figure:
    """Heatmap of ``sign · M<component>`` over time and offset."""
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of 'x', 'y', 'z', got {component!r}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    else:
        fig = ax.figure
    t_factor, t_unit = si_scale(traj.t[-1], "s")
    o_factor, o_unit = si_scale(np.abs(traj.offsets).max(), "Hz")
    data = sign * traj.m[:, :, _COMPONENTS[component]].T
    mesh = ax.pcolormesh(
        traj.t * t_factor,
        traj.offsets * o_factor,
        data,
        cmap="coolwarm",  # diverging blue↔red, neutral gray at 0
        vmin=-1,
        vmax=1,
        shading="nearest",
    )
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label(f"{'−' if sign < 0 else ''}M{component}", color=_INK)
    bar.ax.tick_params(colors=_INK)
    ax.set_xlabel(f"t ({t_unit})")
    ax.set_ylabel(f"offset ({o_unit})")
    _style(ax)
    ax.grid(False)
    return fig


def plot_b1_map(
    bmap: B1Map,
    component: Literal["x", "y", "z"] = "y",
    *,
    sign: float = 1,
    surface: bool = False,
    ax: Axes | None = None,
) -> Figure:
    """``sign · M<component>`` over offset/Ω1(0) and B1 scale Ω1/Ω1(0).

    Filled contours by default; ``surface=True`` draws a 3D
    surface instead (``ax``, if given, must then be a 3D axes).
    """
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of 'x', 'y', 'z', got {component!r}")
    x = bmap.offsets / bmap.rf_ref
    data = sign * bmap.m[:, :, _COMPONENTS[component]]
    label = f"{'−' if sign < 0 else ''}M{component}"
    if surface:
        if ax is None:
            fig = plt.figure(figsize=(7, 5), layout="constrained")
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.figure
        grid_x, grid_y = np.meshgrid(x, bmap.scales)
        ax.plot_surface(
            grid_x,
            grid_y,
            data,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidth=0,
            rcount=data.shape[0],  # every sample (matplotlib's default caps at 50)
            ccount=data.shape[1],
        )
        ax.set_zlim(-1, 1)
        ax.set_zlabel(label)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        else:
            fig = ax.figure
        filled = ax.contourf(
            x,
            bmap.scales,
            data,
            levels=np.linspace(-1, 1, 21),
            cmap="coolwarm",  # diverging blue↔red, neutral gray at 0
            extend="both",
        )
        bar = fig.colorbar(filled, ax=ax)
        bar.set_label(label, color=_INK)
        bar.ax.tick_params(colors=_INK)
        _style(ax)
        ax.grid(False)
    ax.set_xlabel("Δ / Ω1(0)")
    ax.set_ylabel("Ω1 / Ω1(0)")
    return fig


def plot_b1_phase(
    scales: ArrayLike, dphi: ArrayLike, *, delta: float = 0.005, ax: Axes | None = None
) -> Figure:
    """Phase sensitivity from :func:`~nutate.b1.b1_phase_sensitivity` vs B1 scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    else:
        fig = ax.figure
    ax.plot(scales, dphi, color=_LINE, linewidth=1.5)
    ax.set_xlabel("Ω1 / Ω1(0)")
    ax.set_ylabel(f"ΔΦ for ±{delta * 100:g} % B1 (°)")
    _style(ax)
    return fig
