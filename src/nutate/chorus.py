"""CHORUS broadband excitation: 90° chirp, 180° chirp, delay, 180° chirp.

The same function serves NMR (µs pulses, kHz bandwidths) and compressed EPR versions
(ns pulses, MHz bandwidths): only units and step size differ.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike

from nutate.pulses import Q90, Segment, chirp, delay
from nutate.simulate import PhaseCycle, simulate_final

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


def phase_correction(
    sequence: list[Segment],
    bandwidth: float,
    *,
    phase_cycle: PhaseCycle | None = None,
    smoothing: float = 5.0,
    degree: int = 10,
    n_offsets: int = 101,
) -> np.ndarray:
    """Phase correction (degrees, one per step of ``sequence[0]``).

    Simulates ``sequence`` across ``±bandwidth/2`` (optionally phase-cycled), fits a
    polynomial of ``degree`` to the excited phase (excluding ``smoothing`` % of offsets at
    each edge) against offset index, and maps it onto the first segment's steps — valid
    when that segment is a linear chirp, which visits offsets linearly in time. Add the
    result to the first chirp via ``phase_correction_deg``.
    """
    offsets = np.linspace(-bandwidth / 2, bandwidth / 2, n_offsets)
    final = simulate_final(sequence, offsets, phase_cycle=phase_cycle)
    phase = np.rad2deg(np.unwrap(np.angle(final[:, 1] + 1j * final[:, 0]))) + 180
    edge = round(n_offsets * smoothing / 100)
    index = np.arange(edge, n_offsets - edge)
    fit = Polynomial.fit(index + 1, phase[index], degree)
    return fit(np.linspace(1, n_offsets, sequence[0].n_steps))


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
