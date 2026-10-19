"""CHORUS broadband excitation: 90° chirp, 180° chirp, delay, 180° chirp.

The same function serves NMR (µs pulses, kHz bandwidths) and compressed EPR versions
(ns pulses, MHz bandwidths): only units and step size differ.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from nutate.pulses.swept_frequency.chirp import Q90, chirp
from nutate.pulses.swept_frequency.phase_correction import phase_correction
from nutate.segment import Segment, delay
from nutate.simulate import PhaseCycle

# 16-step cycle: phases of the three chirps (the delay column is 0) and the receiver.
CHORUS_CYCLE = PhaseCycle(
    90.0
    * np.column_stack(
        [
            np.zeros(16),
            np.tile([0, 1, 2, 3], 4),
            np.zeros(16),
            np.repeat([0, 1, 2, 3], 4),
        ]
    ),
    90.0 * np.array([0, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0]),
)


def _n_steps(duration: float, dt: float) -> int:
    # The tolerance absorbs float division error (10e-3 / 2e-5 = 499.99999999999994).
    return int(np.floor(duration / dt + 1e-9))


def chorus(
    t90: float,
    t180: float,
    bandwidth: float,
    dt: float,
    *,
    smoothing: float = 5.0,
    q180: float = 5.0,
    phase_correction_deg: ArrayLike | None = None,
) -> list[Segment]:
    """CHORUS sequence for ``bandwidth`` Hz, sampled every ``dt`` s.

    Segments: 90° chirp (``t90``), 180° chirp (``t180 + t90/2``), delay (``t90/2``),
    180° chirp (``t180``), with ``n1 = floor(t90/dt)``, ``n3 = floor(t180/dt)`` and
    ``n2 = n3 + n1 // 2`` steps.
    ``phase_correction_deg`` (one value per step of the first chirp) is added to the
    first chirp only; see :func:`chorus_phase_correction`.
    """
    n1 = _n_steps(t90, dt)
    n3 = _n_steps(t180, dt)
    n2 = n3 + n1 // 2
    return [
        chirp(
            t90,
            bandwidth,
            n1,
            smoothing=smoothing,
            q=Q90,
            phase_correction_deg=phase_correction_deg,
            name="90° chirp",
        ),
        chirp(t180 + t90 / 2, bandwidth, n2, smoothing=smoothing, q=q180, name="180° chirp"),
        delay(n1 // 2 * t90 / n1, n1 // 2),
        chirp(t180, bandwidth, n3, smoothing=smoothing, q=q180, name="180° chirp"),
    ]


def chorus_phase_correction(
    t90: float,
    t180: float,
    bandwidth: float,
    dt: float,
    *,
    smoothing: float = 5.0,
    q180: float = 5.0,
    degree: int = 10,
    n_offsets: int = 101,
) -> np.ndarray:
    """:func:`phase_correction` for :func:`chorus` with :data:`CHORUS_CYCLE`.

    Pass the same ``smoothing`` and ``q180`` as the :func:`chorus` call the correction is for.
    """
    sequence = chorus(t90, t180, bandwidth, dt, smoothing=smoothing, q180=q180)
    return phase_correction(
        sequence,
        bandwidth,
        phase_cycle=CHORUS_CYCLE,
        smoothing=smoothing,
        degree=degree,
        n_offsets=n_offsets,
    )
