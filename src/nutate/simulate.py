"""Rotating-frame Bloch simulation (no relaxation).

Each step rotates the magnetization about the effective field,
``Rz(φ)·Ry(θ)·Rz(α)·Ry(−θ)·Rz(−φ)`` with ``θ = atan2(Ω, Δ)`` and
``α = 2π·sqrt(Ω² + Δ²)·dt``, i.e. a right-handed rotation by ``+α`` about
``n = (sinθ cosφ, sinθ sinφ, cosθ)``. All offsets are rotated at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from nutate.pulses import Segment


@dataclass(frozen=True, eq=False)
class PhaseCycle:
    """Phase cycle: per cycle step, a phase added to each segment and a receiver phase.

    Attributes
    ----------
    pulse_phases_deg : shape ``(n_cycle, n_segments)``, degrees added to each segment's phase.
    receiver_deg : shape ``(n_cycle,)``, receiver phase in degrees; each cycle step's
        magnetization is rotated by ``Rz(−receiver)`` before averaging.
    """

    pulse_phases_deg: np.ndarray
    receiver_deg: np.ndarray

    def __post_init__(self) -> None:
        pulse = np.asarray(self.pulse_phases_deg, dtype=float)
        receiver = np.asarray(self.receiver_deg, dtype=float)
        if pulse.ndim != 2 or pulse.shape[0] == 0:
            raise ValueError(
                f"pulse_phases_deg must have shape (n_cycle, n_segments), got {pulse.shape}"
            )
        if receiver.shape != (pulse.shape[0],):
            raise ValueError(
                f"receiver_deg must have shape ({pulse.shape[0]},), got {receiver.shape}"
            )
        object.__setattr__(self, "pulse_phases_deg", pulse)
        object.__setattr__(self, "receiver_deg", receiver)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Magnetization over time for every offset.

    Attributes
    ----------
    t : times, s, shape ``(n_steps + 1,)``; ``t[0] = 0``.
    offsets : offsets, Hz, shape ``(n_off,)``.
    m : magnetization, shape ``(n_steps + 1, n_off, 3)``; ``m[0]`` is the initial state.
    segments : ``(name, start, stop)`` per segment; ``t[start]`` and ``t[stop]`` are its
        start and end, its states are ``m[start:stop + 1]``.
    sequence : the segments simulated.
    """

    t: np.ndarray
    offsets: np.ndarray
    m: np.ndarray
    segments: list[tuple[str, int, int]]
    sequence: list[Segment]

    @property
    def final(self) -> np.ndarray:
        """Magnetization after the last step, shape ``(n_off, 3)``."""
        return self.m[-1]


def _rotate(
    m: np.ndarray, amp: float, phase: np.ndarray | float, offsets: np.ndarray, dt: float
) -> np.ndarray:
    """Apply one step to ``m`` of shape ``(..., n_off, 3)``.

    ``amp`` in Hz, ``phase`` in rad (scalar or broadcastable to ``(..., n_off)``),
    ``offsets`` in Hz, ``dt`` in s. Rodrigues' formula:
    ``M' = M cosα + (n × M) sinα + n (n·M)(1 − cosα)``.
    """
    alpha = 2 * np.pi * np.hypot(amp, offsets) * dt
    theta = np.arctan2(amp, offsets)
    sin_t, nz = np.sin(theta), np.cos(theta)
    nx, ny = sin_t * np.cos(phase), sin_t * np.sin(phase)
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    ca, sa = np.cos(alpha), np.sin(alpha)
    k = (nx * mx + ny * my + nz * mz) * (1 - ca)
    return np.stack(
        [
            mx * ca + (ny * mz - nz * my) * sa + nx * k,
            my * ca + (nz * mx - nx * mz) * sa + ny * k,
            mz * ca + (nx * my - ny * mx) * sa + nz * k,
        ],
        axis=-1,
    )


def _check_offsets(offsets: ArrayLike) -> np.ndarray:
    offsets = np.atleast_1d(np.asarray(offsets, dtype=float))
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError(f"offsets must be a non-empty 1-D array, got shape {offsets.shape}")
    if not np.isfinite(offsets).all():
        raise ValueError("offsets must be finite")
    return offsets


def _segment_offsets(segment: Segment, offsets: np.ndarray) -> np.ndarray:
    if segment.offset_fn is None:
        return offsets
    mapped = np.asarray(segment.offset_fn(offsets), dtype=float)
    if mapped.shape != offsets.shape:
        raise ValueError(f"offset_fn of segment {segment.name!r} changed the offsets' shape")
    return mapped


def _check_b1(b1: ArrayLike, offsets: np.ndarray) -> np.ndarray | float:
    b1 = np.asarray(b1, dtype=float)
    if b1.ndim and b1.shape != offsets.shape:
        raise ValueError(f"b1 must be a scalar or match offsets {offsets.shape}, got {b1.shape}")
    if not np.isfinite(b1).all() or (b1 < 0).any():
        raise ValueError("b1 must be finite and ≥ 0")
    return float(b1) if b1.ndim == 0 else b1


def _propagate(
    sequence: Sequence[Segment],
    offsets: ArrayLike,
    *,
    m0: ArrayLike,
    phase_cycle: PhaseCycle | None,
    b1: ArrayLike,
    record: bool,
):
    """Shared loop: the full :class:`Trajectory` fields if ``record``, else the final state."""
    sequence = list(sequence)
    if not sequence:
        raise ValueError("sequence must contain at least one Segment")
    offsets = _check_offsets(offsets)
    b1 = _check_b1(b1, offsets)
    m0 = np.asarray(m0, dtype=float)
    if m0.shape != (3,):
        raise ValueError(f"m0 must have 3 components, got shape {m0.shape}")
    if phase_cycle is None:
        phase_cycle = PhaseCycle(np.zeros((1, len(sequence))), np.zeros(1))
    elif phase_cycle.pulse_phases_deg.shape[1] != len(sequence):
        raise ValueError(
            f"phase_cycle has {phase_cycle.pulse_phases_deg.shape[1]} segment columns, "
            f"sequence has {len(sequence)} segments"
        )
    # Shapes (n_cycle, 1) so they broadcast against (n_cycle, n_off).
    pulse_phases = np.deg2rad(phase_cycle.pulse_phases_deg)[:, :, None]
    receiver = -np.deg2rad(phase_cycle.receiver_deg)[:, None]
    rec_cos, rec_sin = np.cos(receiver), np.sin(receiver)

    def observed(batch: np.ndarray) -> np.ndarray:
        x, y = batch[..., 0], batch[..., 1]
        rotated = np.stack(
            [x * rec_cos - y * rec_sin, x * rec_sin + y * rec_cos, batch[..., 2]], -1
        )
        return rotated.mean(axis=0)

    current = np.broadcast_to(m0, (receiver.shape[0], offsets.size, 3)).copy()
    if record:
        dts = np.concatenate([np.full(s.n_steps, s.dt) for s in sequence])
        t = np.concatenate([[0.0], np.cumsum(dts)])
        m = np.empty((t.size, offsets.size, 3))
        m[0] = observed(current)
    segments = []
    k = 0
    for i, seg in enumerate(sequence):
        offs = _segment_offsets(seg, offsets)
        segments.append((seg.name, k, k + seg.n_steps))
        for amp, phase in zip(seg.amp, seg.phase):
            current = _rotate(current, amp * b1, phase + pulse_phases[:, i], offs, seg.dt)
            k += 1
            if record:
                m[k] = observed(current)
    if record:
        return t, offsets, m, segments, sequence
    return observed(current)


def simulate(
    sequence: Sequence[Segment],
    offsets: ArrayLike,
    *,
    m0: ArrayLike = (0, 0, 1),
    phase_cycle: PhaseCycle | None = None,
    b1: ArrayLike = 1.0,
) -> Trajectory:
    """Simulate ``sequence`` for each offset (Hz), starting from ``m0``.

    With ``phase_cycle``, every cycle step is simulated (as one batch), rotated by its
    receiver phase, and the returned trajectory is the mean over the cycle. ``b1`` scales
    every RF amplitude: a scalar, or one factor per offset.
    """
    return Trajectory(
        *_propagate(sequence, offsets, m0=m0, phase_cycle=phase_cycle, b1=b1, record=True)
    )


def simulate_final(
    sequence: Sequence[Segment],
    offsets: ArrayLike,
    *,
    m0: ArrayLike = (0, 0, 1),
    phase_cycle: PhaseCycle | None = None,
    b1: ArrayLike = 1.0,
) -> np.ndarray:
    """Final magnetization only, shape ``(n_off, 3)``; same options as :func:`simulate`.

    Stores no trajectory, so large offset × B1 grids stay small in memory.
    """
    return _propagate(sequence, offsets, m0=m0, phase_cycle=phase_cycle, b1=b1, record=False)
