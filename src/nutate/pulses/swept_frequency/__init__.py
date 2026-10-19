"""Swept-frequency pulses: chirps, hyperbolic-secant pulses, CHORUS and phase correction."""

from nutate.pulses.swept_frequency.chirp import Q90, chirp, q_factor
from nutate.pulses.swept_frequency.chorus import CHORUS_CYCLE, chorus, chorus_phase_correction
from nutate.pulses.swept_frequency.hs import hs
from nutate.pulses.swept_frequency.phase_correction import phase_correction

__all__ = [
    "CHORUS_CYCLE",
    "Q90",
    "chirp",
    "chorus",
    "chorus_phase_correction",
    "hs",
    "phase_correction",
    "q_factor",
]
