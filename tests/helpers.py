"""Test pulses built through the public API."""

import numpy as np

from nutate import Segment, waveform


def hard(
    duration: float, *, flip_deg: float = 90.0, phase_deg: float = 0.0, n_steps: int = 1000
) -> Segment:
    """Rectangular pulse of ``duration`` s giving ``flip_deg`` on resonance."""
    return waveform(
        np.ones(n_steps),
        phase_deg=np.full(n_steps, phase_deg),
        duration=duration,
        rabi=flip_deg / (360 * duration),
        name="hard",
    )
