"""Linear frequency-swept (chirp) pulses."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from nutate.segment import Segment


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
    shaped by a quarter-sine ramp. Rabi frequency ``sqrt(duration·bandwidth·q/(2π))/duration``
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
    rabi = np.sqrt(duration * bandwidth * q / (2 * np.pi)) / duration
    return Segment(name, envelope, phase, duration / n_steps, rabi)
