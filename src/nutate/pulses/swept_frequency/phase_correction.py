"""Phase correction for sequences that start with a linear chirp."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial

from nutate.segment import Segment
from nutate.simulate import PhaseCycle, simulate_final


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
