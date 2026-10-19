"""The pulse type and its generic constructors.

Every pulse or delay is a :class:`Segment`: a waveform normalised to 1 and its Rabi
frequency, i.e. a train of piecewise-constant steps, each with an amplitude (in units of
the Rabi frequency) and RF phase (rad) held for ``dt`` seconds. :func:`waveform` makes one
from your own arrays, :func:`delay` adds free precession. Pulse families (swept-frequency,
composite) live in :mod:`nutate.pulses`.
A sequence is a plain ``list[Segment]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, eq=False)
class Segment:
    """One pulse or delay, sampled in equal steps.

    Parameters
    ----------
    name : label shown in videos (e.g. ``"90° chirp"``).
    amp : waveform per step, normalised to 1 (≥ 0; a peak above 1.5 is an error).
    phase : RF phase per step, rad.
    dt : step duration, s, > 0.
    rabi : Rabi frequency ν1, Hz, at waveform value 1 (≥ 0); the RF amplitude is
        ``amp · rabi``.
    offset_fn : optional map applied to the offsets (Hz) while this segment runs;
        ``None`` keeps them unchanged.
    """

    name: str
    amp: np.ndarray
    phase: np.ndarray
    dt: float
    rabi: float
    offset_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        amp = np.asarray(self.amp, dtype=float)
        phase = np.asarray(self.phase, dtype=float)
        if amp.ndim != 1 or amp.size == 0:
            raise ValueError(f"amp must be a non-empty 1-D array, got shape {amp.shape}")
        if phase.shape != amp.shape:
            raise ValueError(f"phase shape {phase.shape} must match amp shape {amp.shape}")
        if not np.isfinite(amp).all() or (amp < 0).any():
            raise ValueError("amp must be finite and ≥ 0")
        if amp.max() > 1.5:
            raise ValueError(
                f"amp peak is {amp.max():g}; it must be normalised to 1 (rabi sets the Hz)"
            )
        if not np.isfinite(phase).all():
            raise ValueError("phase must be finite")
        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be a positive number of seconds, got {self.dt}")
        if not (np.isfinite(self.rabi) and self.rabi >= 0):
            raise ValueError(f"rabi must be a frequency ≥ 0 in Hz, got {self.rabi}")
        object.__setattr__(self, "amp", amp)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "rabi", float(self.rabi))

    @property
    def n_steps(self) -> int:
        return self.amp.size

    @property
    def duration(self) -> float:
        return self.n_steps * self.dt

    @property
    def rf(self) -> np.ndarray:
        """RF amplitude per step, Hz: ``amp · rabi``."""
        return self.amp * self.rabi

    def with_rabi(self, rabi: float) -> Segment:
        """Copy with Rabi frequency ``rabi`` Hz (waveform kept)."""
        return replace(self, rabi=rabi)


def delay(
    duration: float,
    n_steps: int,
    *,
    name: str = "delay",
    offset_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Segment:
    """Free precession for ``duration`` s in ``n_steps`` equal steps (RF off, ``rabi = 0``)."""
    return Segment(name, np.zeros(n_steps), np.zeros(n_steps), duration / n_steps, 0.0, offset_fn)


def _paired(name_a: str, a: ArrayLike, name_b: str, b: ArrayLike) -> tuple[np.ndarray, ...]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {a.shape} and {b.shape}"
        )
    return a, b


def waveform(
    amp: ArrayLike | None = None,
    phase: ArrayLike | None = None,
    *,
    phase_deg: ArrayLike | None = None,
    i: ArrayLike | None = None,
    q: ArrayLike | None = None,
    duration: float,
    rabi: float | None = None,
    t90: float | None = None,
    name: str = "waveform",
) -> Segment:
    """Pulse of ``duration`` s from a waveform normalised to 1, one step per sample.

    Give ``amp`` with ``phase`` (rad) or ``phase_deg``, or ``i`` and ``q`` (I along x, Q along
    y: ``amp = |I + iQ|``, ``phase = angle(I + iQ)``). ``rabi`` is the Rabi frequency, Hz, at
    waveform value 1; or give ``t90``, the hard 90° pulse length at that amplitude, and
    ``rabi = 1 / (4·t90)``. The waveform is not rescaled: a slight overshoot above 1 is
    simulated as given, a peak above 1.5 is an error. A negative ``amp`` is taken as ``|amp|``
    with the phase shifted by π.
    """
    use_iq = i is not None or q is not None
    if use_iq == (amp is not None or phase is not None or phase_deg is not None):
        raise ValueError("give either amp with phase (or phase_deg), or i and q")
    if use_iq:
        if i is None or q is None:
            raise ValueError("i and q must be given together")
        i, q = _paired("i", i, "q", q)
        amp, phase = np.hypot(i, q), np.arctan2(q, i)
    else:
        if amp is None:
            raise ValueError("amp is needed with phase")
        if (phase is None) == (phase_deg is None):
            raise ValueError("give exactly one of phase (rad) or phase_deg")
        amp, phase = _paired("amp", amp, "phase", np.deg2rad(phase_deg) if phase is None else phase)
        phase = np.where(amp < 0, phase + np.pi, phase)
        amp = np.abs(amp)
    if amp.ndim != 1 or amp.size == 0:
        raise ValueError(f"waveform must be a non-empty 1-D array, got shape {amp.shape}")
    if (rabi is None) == (t90 is None):
        raise ValueError("give exactly one of rabi (Hz) or t90 (s)")
    if t90 is not None:
        if not (np.isfinite(t90) and t90 > 0):
            raise ValueError(f"t90 must be a positive number of seconds, got {t90}")
        rabi = 1 / (4 * t90)
    elif not (np.isfinite(rabi) and rabi > 0):
        raise ValueError(f"rabi must be a positive frequency in Hz, got {rabi}")
    return Segment(name, amp, phase, duration / amp.size, rabi)


__all__ = ["Segment", "delay", "waveform"]
