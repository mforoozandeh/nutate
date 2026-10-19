"""Bloch simulations of any pulse sequence rendered as 3D Bloch-sphere videos."""

from importlib.metadata import version

from nutate.b1 import B1Map, b1_map, b1_phase_sensitivity
from nutate.profiles import plot_b1_map, plot_b1_phase, plot_map, plot_profiles
from nutate.render import Camera, Style, render, show, snapshot
from nutate.run import RunResult, run
from nutate.segment import Segment, delay, waveform
from nutate.simulate import PhaseCycle, Trajectory, simulate, simulate_final

__version__ = version("nutate")

__all__ = [
    "B1Map",
    "Camera",
    "PhaseCycle",
    "RunResult",
    "Segment",
    "Style",
    "Trajectory",
    "b1_map",
    "b1_phase_sensitivity",
    "delay",
    "plot_b1_map",
    "plot_b1_phase",
    "plot_map",
    "plot_profiles",
    "render",
    "run",
    "show",
    "simulate",
    "simulate_final",
    "snapshot",
    "waveform",
]
