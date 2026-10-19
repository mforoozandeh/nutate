"""B1 robustness of ABSTRUSE (hyperbolic-secant CHORUS analogue).

HS pulses of T, T and T/2 (βT = 10.6) at Rabi frequencies k·sqrt(bandwidth/duration),
k = 0.61, 1.83, 1.93, the last two at 90° phase, a T/2 delay before the last, 16-step cycle;
offsets normalised by the third pulse's Rabi frequency. Writes the +My map, its surface and
the on-resonance phase sensitivity. Illustrative parameters: T = 1 ms, 300 kHz, 1000 steps.
"""

import numpy as np

import nutate as nt
from nutate.pulses.swept_frequency import CHORUS_CYCLE, hs

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    rf = [k * np.sqrt(BANDWIDTH / d) for k, d in ((0.61, T), (1.83, T), (1.93, T / 2))]
    sequence = [
        hs(T, BANDWIDTH, N, rabi=rf[0]),
        hs(T, BANDWIDTH, N, rabi=rf[1], phase_deg=90),
        nt.delay(T / 2, N // 2),
        hs(T / 2, BANDWIDTH, N // 2, rabi=rf[2], phase_deg=90),
    ]
    result = nt.run(
        sequence,
        BANDWIDTH,
        name="b1_abstruse",
        video=False,
        b1=True,
        sign=1,
        phase_cycle=CHORUS_CYCLE,
        out="output",
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
