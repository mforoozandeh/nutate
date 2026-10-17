"""B1 robustness of ABSTRUSE (hyperbolic-secant CHORUS analogue).

HS pulses of T, T and T/2 (βT = 10.6) with peaks k·sqrt(bandwidth/duration), k = 0.61, 1.83,
1.93, the last two at 90° phase, a T/2 delay before the last, 16-step cycle; offsets
normalised by the third pulse's peak. Writes the +My map, its surface and the on-resonance
phase sensitivity. Illustrative parameters: T = 1 ms, 300 kHz, 1000 steps.
"""

import numpy as np
from _common import b1_axes, finish_b1, parse_args

import nutate as nt

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    args = parse_args(__doc__)
    rf = [k * np.sqrt(BANDWIDTH / d) for k, d in ((0.61, T), (1.83, T), (1.93, T / 2))]
    sequence = [
        nt.hs(T, BANDWIDTH, N, rf_max_hz=rf[0]),
        nt.hs(T, BANDWIDTH, N, rf_max_hz=rf[1], phase_deg=90),
        nt.delay(T / 2, N // 2),
        nt.hs(T / 2, BANDWIDTH, N // 2, rf_max_hz=rf[2], phase_deg=90),
    ]
    offsets, scales = b1_axes(args, BANDWIDTH)
    finish_b1(
        "b1_abstruse",
        args,
        sequence,
        offsets,
        scales,
        rf_ref=rf[2],
        sign=1,
        phase_cycle=nt.CHORUS_CYCLE,
    )


if __name__ == "__main__":
    main()
