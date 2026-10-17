"""Pulse sequence building blocks.

Every pulse or delay is a :class:`Segment`: a train of piecewise-constant steps, each
with an RF amplitude (Hz) and RF phase (rad) held for ``dt`` seconds. A sequence is a
plain ``list[Segment]``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, eq=False)
class Segment:
    """One pulse or delay, sampled in equal steps.

    Parameters
    ----------
    name : label shown in videos (e.g. ``"90° chirp"``).
    amp : RF amplitude per step, Hz, ≥ 0.
    phase : RF phase per step, rad.
    dt : step duration, s, > 0.
    offset_fn : optional map applied to the offsets (Hz) while this segment runs;
        ``None`` keeps them unchanged.
    """

    name: str
    amp: np.ndarray
    phase: np.ndarray
    dt: float
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
        if not np.isfinite(phase).all():
            raise ValueError("phase must be finite")
        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be a positive number of seconds, got {self.dt}")
        object.__setattr__(self, "amp", amp)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "dt", float(self.dt))

    @property
    def n_steps(self) -> int:
        return self.amp.size

    @property
    def duration(self) -> float:
        return self.n_steps * self.dt

    def with_peak(self, peak_hz: float) -> Segment:
        """Copy with the amplitude rescaled so its maximum is ``peak_hz`` (shape kept)."""
        top = self.amp.max()
        if top == 0:
            raise ValueError(f"segment {self.name!r} has no RF (amp is all zero) to rescale")
        return Segment(self.name, self.amp * (peak_hz / top), self.phase, self.dt, self.offset_fn)


def hard(
    duration: float,
    *,
    flip_deg: float = 90.0,
    phase_deg: float = 0.0,
    n_steps: int = 1000,
    name: str = "hard",
) -> Segment:
    """Rectangular pulse of ``duration`` s giving ``flip_deg`` on resonance.

    Amplitude is ``flip_deg / (360 · duration)`` Hz, split into ``n_steps`` equal steps.
    """
    amp = flip_deg / (360.0 * duration)
    return Segment(
        name,
        np.full(n_steps, amp),
        np.full(n_steps, np.deg2rad(phase_deg)),
        duration / n_steps,
    )


def delay(
    duration: float,
    n_steps: int,
    *,
    name: str = "delay",
    offset_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Segment:
    """Free precession for ``duration`` s in ``n_steps`` equal steps (RF off)."""
    return Segment(name, np.zeros(n_steps), np.zeros(n_steps), duration / n_steps, offset_fn)


def q_factor(flip_deg: float) -> float:
    """Adiabaticity factor giving ``flip_deg`` with a linear chirp.

    ``Q = −2·ln(cos(flip)/2 + 1/2)/π``, valid for ``0 < flip_deg < 180``
    (``q_factor(90) ≈ 0.4413``). Inversion and refocusing chirps use a fixed ``q``,
    typically 5.
    """
    if not 0 < flip_deg < 180:
        raise ValueError(
            f"q_factor needs 0 < flip_deg < 180, got {flip_deg}; "
            "for inversion/refocusing pass q=5 to chirp() instead"
        )
    return float(-2 * np.log(np.cos(np.deg2rad(flip_deg)) / 2 + 0.5) / np.pi)


Q90 = q_factor(90)


def chirp(
    duration: float,
    bandwidth: float,
    n_steps: int,
    *,
    smoothing: float = 10.0,
    q: float = Q90,
    phase_correction_deg: ArrayLike | None = None,
    name: str = "chirp",
) -> Segment:
    """Linear frequency sweep from ``−bandwidth/2`` to ``+bandwidth/2`` Hz.

    ``duration`` in s, ``n_steps`` samples, ``smoothing`` = % of samples at each end
    shaped by a quarter-sine ramp. Peak amplitude ``sqrt(duration·bandwidth·q/(2π))/duration``
    Hz. ``phase_correction_deg`` (one value per step, degrees) is added to the phase.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be ≥ 2, got {n_steps}")
    if not 0 <= smoothing < 50:
        raise ValueError(f"smoothing must be in [0, 50) %, got {smoothing}")
    # Phase sampled on linspace(0, T, n) while each step lasts T/n.
    t = np.linspace(0, duration, n_steps)
    phase = 2 * np.pi * (bandwidth / t[-1] * t**2 / 2 - bandwidth / 2 * t)
    phase -= phase.min()
    if phase_correction_deg is not None:
        correction = np.asarray(phase_correction_deg, dtype=float)
        if correction.shape != (n_steps,):
            raise ValueError(
                f"phase_correction_deg must have shape ({n_steps},), got {correction.shape}"
            )
        phase += np.deg2rad(correction)

    n_ramp = int(np.floor(n_steps * smoothing / 100 + 1e-9))
    envelope = np.ones(n_steps)
    if n_ramp:
        ramp = np.sin(np.linspace(0, np.pi / 2, n_ramp))
        envelope[:n_ramp] = ramp
        envelope[-n_ramp:] = ramp[::-1]
    peak = np.sqrt(duration * bandwidth * q / (2 * np.pi)) / duration
    return Segment(name, peak * envelope, phase, duration / n_steps)


def hs(
    duration: float,
    bandwidth: float,
    n_steps: int,
    *,
    rf_max_hz: float,
    phase_deg: float = 0.0,
    truncation: float = 10.6,
    name: str = "HS pulse",
) -> Segment:
    """Hyperbolic-secant pulse sweeping ``±bandwidth/2`` Hz over ``duration`` s.

    Amplitude ``rf_max_hz · sech(β·t)`` and phase ``phase_deg + (π·bandwidth/β)·ln cosh(β·t)``
    for ``t`` in ``[−duration/2, duration/2]``, with ``β = truncation/duration`` (the pulse is
    cut at ``sech(truncation/2)``).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be ≥ 2, got {n_steps}")
    t = np.linspace(-duration / 2, duration / 2, n_steps)
    beta = truncation / duration
    phase = np.deg2rad(phase_deg) + np.pi * bandwidth / beta * np.log(np.cosh(beta * t))
    return Segment(name, rf_max_hz / np.cosh(beta * t), phase, duration / n_steps)


# (flips_deg, phases_deg) composite-pulse presets, keyed by number ("10", a duplicate
# of "4", is omitted).
COMPOSITE_PRESETS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "1": ((90,), (0,)),
    "2": ((90, 90), (0, 90)),
    "3": ((90, 180), (0, 120)),
    "4": ((180, 360, 180, 90), (97.2, 291.5, 97.2, 0)),
    "5": ((180, 360, 180, 270, 90), (0, 180, 0, 180, 90)),
    "6": ((385, 320, 25), (0, 180, 0)),
    "7": ((113, 316, 113), (180, 0, 180)),
    "8": ((24, 152, 346, 152, 24), (0, 180, 0, 180, 0)),
    "9": ((119, 183, 211, 384, 211, 183, 119), (180, 0, 180, 0, 180, 0, 180)),
    "11": ((180, 360, 180, 270, 90, 360, 270, 90), (0, 180, 0, 180, 90, 0, 180, 90)),
    "12": ((90, 180, 90), (0, 90, 0)),
    "13": ((180,), (0,)),
    "14": ((180, 360, 180, 180), (104.5, 313.4, 104.5, 0)),
    "15": ((180, 180, 180), (120, 240, 120)),
    "16": ((1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144), (180, 0) * 6),
}


def composite(
    t90: float,
    flips_deg: Sequence[float],
    phases_deg: Sequence[float],
    *,
    name: str = "composite",
) -> list[Segment]:
    """Composite pulse of rectangular elements at ``1/(4·t90)`` Hz, ``t90`` in s.

    Element ``i`` flips by ``flips_deg[i]`` (a positive integer) about phase
    ``phases_deg[i]``, sampled one step per degree (``dt = t90/90``).
    Returns one segment per element.
    """
    if len(flips_deg) != len(phases_deg) or len(flips_deg) == 0:
        raise ValueError("flips_deg and phases_deg must be non-empty and the same length")
    for flip in flips_deg:
        if flip <= 0 or flip != int(flip):
            raise ValueError(f"flips_deg must be positive integers, got {flip}")
    amp = 1 / (4 * t90)
    return [
        Segment(
            f"{name} {flip:g}°({phase:g}°)",
            np.full(int(flip), amp),
            np.full(int(flip), np.deg2rad(phase)),
            t90 / 90,
        )
        for flip, phase in zip(flips_deg, phases_deg)
    ]


def read_bruker_shape(
    path: str | os.PathLike,
    duration: float,
    rf_max_hz: float,
    *,
    name: str | None = None,
) -> Segment:
    """Read a Bruker (JCAMP-DX) shape file as a pulse of ``duration`` s.

    Data lines between ``##XYPOINTS=`` and ``##END=`` hold ``amplitude_percent, phase_deg``.
    Amplitude is scaled so 100 % = ``rf_max_hz``. ``name`` defaults to the file stem.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().upper().startswith("##XYPOINTS")),
        None,
    )
    if start is None:
        raise ValueError(f"{path}: no ##XYPOINTS= data block")
    amp, phase = [], []
    for number, line in enumerate(lines[start + 1 :], start=start + 2):
        text = line.strip()
        if not text or text.startswith("$$"):
            continue
        if text.startswith("##"):
            if text.upper().startswith("##END"):
                break
            continue
        try:
            a, p = (float(v) for v in text.split(","))
        except ValueError:
            raise ValueError(
                f"{path}: line {number}: expected 'amplitude, phase', got {text!r}"
            ) from None
        amp.append(a)
        phase.append(p)
    if not amp:
        raise ValueError(f"{path}: ##XYPOINTS= block has no data lines")
    return Segment(
        path.stem if name is None else name,
        np.asarray(amp) / 100 * rf_max_hz,
        np.deg2rad(phase),
        duration / len(amp),
    )


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
    if amp.max() > 1.5:
        raise ValueError(
            f"waveform peak is {amp.max():g}; it must be normalised to 1 (rabi sets the Hz)"
        )
    if (rabi is None) == (t90 is None):
        raise ValueError("give exactly one of rabi (Hz) or t90 (s)")
    if t90 is not None:
        if not (np.isfinite(t90) and t90 > 0):
            raise ValueError(f"t90 must be a positive number of seconds, got {t90}")
        rabi = 1 / (4 * t90)
    elif not (np.isfinite(rabi) and rabi > 0):
        raise ValueError(f"rabi must be a positive frequency in Hz, got {rabi}")
    return Segment(name, amp * rabi, phase, duration / amp.size)


__all__ = [
    "COMPOSITE_PRESETS",
    "Q90",
    "Segment",
    "chirp",
    "composite",
    "delay",
    "hard",
    "hs",
    "q_factor",
    "read_bruker_shape",
    "waveform",
]
