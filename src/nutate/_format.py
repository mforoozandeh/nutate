"""SI-prefix scaling for axis labels, overlays and colorbars."""

from __future__ import annotations

import math

_PREFIXES = {-9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}


def si_scale(max_abs: float, unit: str) -> tuple[float, str]:
    """Factor and prefixed unit so that ``max_abs · factor`` lies in ``[1, 1000)``.

    ``si_scale(15e3, "Hz") == (1e-3, "kHz")``; zero gives ``(1.0, unit)``. Prefixes run
    from n to G.
    """
    if max_abs == 0 or not math.isfinite(max_abs):
        return 1.0, unit
    exponent = 3 * math.floor(math.log10(abs(max_abs)) / 3 + 1e-9)
    exponent = min(max(exponent, -9), 9)
    return 10.0**-exponent, _PREFIXES[exponent] + unit
