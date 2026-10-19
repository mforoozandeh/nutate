"""Composite pulses: trains of rectangular elements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nutate.segment import Segment

# (flips_deg, phases_deg) composite-pulse presets, keyed by number ("10", a duplicate
# of "4", is omitted).
PRESETS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
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
    """Composite pulse of rectangular elements at Rabi frequency ``1/(4·t90)`` Hz, ``t90`` in s.

    Element ``i`` flips by ``flips_deg[i]`` (a positive integer) about phase
    ``phases_deg[i]``, sampled one step per degree (``dt = t90/90``).
    Returns one segment per element.
    """
    if len(flips_deg) != len(phases_deg) or len(flips_deg) == 0:
        raise ValueError("flips_deg and phases_deg must be non-empty and the same length")
    for flip in flips_deg:
        if flip <= 0 or flip != int(flip):
            raise ValueError(f"flips_deg must be positive integers, got {flip}")
    return [
        Segment(
            f"{name} {flip:g}°({phase:g}°)",
            np.ones(int(flip)),
            np.full(int(flip), np.deg2rad(phase)),
            t90 / 90,
            1 / (4 * t90),
        )
        for flip, phase in zip(flips_deg, phases_deg)
    ]
