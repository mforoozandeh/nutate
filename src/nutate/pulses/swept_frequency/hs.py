"""Hyperbolic-secant pulses."""

from __future__ import annotations

import numpy as np

from nutate.segment import Segment


def hs(
    duration: float,
    bandwidth: float,
    n_steps: int,
    *,
    rabi: float,
    phase_deg: float = 0.0,
    truncation: float = 10.6,
    name: str = "HS pulse",
) -> Segment:
    """Hyperbolic-secant pulse sweeping ``±bandwidth/2`` Hz over ``duration`` s.

    Waveform ``sech(β·t)`` at Rabi frequency ``rabi`` Hz and phase ``phase_deg + (π·bandwidth/β)·ln cosh(β·t)``
    for ``t`` in ``[−duration/2, duration/2]``, with ``β = truncation/duration`` (the pulse is
    cut at ``sech(truncation/2)``).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be ≥ 2, got {n_steps}")
    t = np.linspace(-duration / 2, duration / 2, n_steps)
    beta = truncation / duration
    phase = np.deg2rad(phase_deg) + np.pi * bandwidth / beta * np.log(np.cosh(beta * t))
    return Segment(name, 1 / np.cosh(beta * t), phase, duration / n_steps, rabi)
