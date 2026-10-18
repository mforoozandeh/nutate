"""B1 robustness of a double-chirp excitation.

A chirp of duration T at 1/4.75 of the Q = 5 Rabi frequency for T/2, a T/2 chirp at that
Rabi frequency, then a T/2 delay; 4-step cycle on the second chirp. Writes the +My map, its
surface and the on-resonance phase sensitivity. Illustrative parameters: T = 1 ms, 300 kHz,
1000 steps.
"""

import numpy as np
from _common import parse_args, report, run_options

import nutate as nt

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    args = parse_args(__doc__)
    rabi = np.sqrt(T / 2 * BANDWIDTH * 5 / (2 * np.pi)) / (T / 2)
    sequence = [
        nt.chirp(T, BANDWIDTH, N, smoothing=5).with_rabi(rabi / 4.75),
        nt.chirp(T / 2, BANDWIDTH, N // 2, smoothing=5).with_rabi(rabi),
        nt.delay(T / 2, N // 2),
    ]
    cycle = nt.PhaseCycle(
        np.column_stack([np.zeros(4), 90.0 * np.arange(4), np.zeros(4)]),
        90.0 * np.array([0, 2, 0, 2]),
    )
    result = nt.run(
        sequence,
        BANDWIDTH,
        name="b1_double_chirp",
        video=False,
        b1=True,
        sign=1,
        phase_cycle=cycle,
        **run_options(args),
    )
    report(result)


if __name__ == "__main__":
    main()
