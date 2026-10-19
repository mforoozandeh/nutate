"""Robustness to RF (B1) amplitude errors: offset × B1 maps and phase sensitivity.

A B1 scale ``z`` multiplies every RF amplitude of the sequence, i.e. ``Ω1 / Ω1(0)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from nutate.segment import Segment
from nutate.simulate import PhaseCycle, _check_offsets, simulate_final


@dataclass(frozen=True, eq=False)
class B1Map:
    """Final magnetization over a grid of offsets and B1 scales.

    Attributes
    ----------
    offsets : offsets, Hz, shape ``(n_off,)``.
    scales : B1 scale factors ``Ω1/Ω1(0)``, shape ``(n_scales,)``.
    m : final magnetization, shape ``(n_scales, n_off, 3)``.
    rf_ref : reference RF amplitude ``Ω1(0)``, Hz, used to normalise offsets in plots.
    """

    offsets: np.ndarray
    scales: np.ndarray
    m: np.ndarray
    rf_ref: float


def _check_scales(scales: ArrayLike) -> np.ndarray:
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError(f"scales must be a non-empty 1-D array, got shape {scales.shape}")
    if not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("scales must be finite and ≥ 0")
    return scales


def b1_map(
    sequence: Sequence[Segment],
    offsets: ArrayLike,
    scales: ArrayLike,
    *,
    rf_ref: float | None = None,
    phase_cycle: PhaseCycle | None = None,
) -> B1Map:
    """Final magnetization for every (B1 scale, offset) pair.

    ``rf_ref`` (Hz) defaults to the sequence's highest Rabi frequency; pass another reference
    (e.g. one pulse's peak) to normalise the offset axis by it.
    """
    sequence = list(sequence)
    offsets = _check_offsets(offsets)
    scales = _check_scales(scales)
    final = simulate_final(
        sequence,
        np.tile(offsets, scales.size),
        phase_cycle=phase_cycle,
        b1=np.repeat(scales, offsets.size),
    )
    if rf_ref is None:
        rf_ref = max(s.rabi for s in sequence)
    return B1Map(offsets, scales, final.reshape(scales.size, offsets.size, 3), rf_ref)


def b1_phase_sensitivity(
    sequence: Sequence[Segment],
    scales: ArrayLike,
    *,
    delta: float = 0.005,
    phase_cycle: PhaseCycle | None = None,
) -> np.ndarray:
    """On-resonance phase change (degrees) for a ``±delta`` B1 change around each scale.

    Phase is ``angle(My + i·Mx)``; the result is ``phase(z·(1+δ)) − phase(z·(1−δ))``
    wrapped to ``[−180, 180)``.
    """
    scales = _check_scales(scales)
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    b1 = np.concatenate([scales * (1 - delta), scales * (1 + delta)])
    final = simulate_final(sequence, np.zeros(b1.size), phase_cycle=phase_cycle, b1=b1)
    phase = np.rad2deg(np.angle(final[:, 1] + 1j * final[:, 0]))
    change = phase[scales.size :] - phase[: scales.size]
    return (change + 180) % 360 - 180
