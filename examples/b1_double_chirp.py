"""B1 robustness of a double-chirp excitation.

A chirp of duration T at 1/4.75 of the Q = 5 peak for T/2, a T/2 chirp at that peak, then a
T/2 delay; 4-step cycle on the second chirp. Writes the +My map, its surface and the
on-resonance phase sensitivity. Illustrative parameters: T = 1 ms, 300 kHz, 1000 steps.
"""

import numpy as np
from _common import b1_axes, finish_b1, parse_args

import nutate as nt

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    args = parse_args(__doc__)
    peak = np.sqrt(T / 2 * BANDWIDTH * 5 / (2 * np.pi)) / (T / 2)
    sequence = [
        nt.chirp(T, BANDWIDTH, N, smoothing=5).with_peak(peak / 4.75),
        nt.chirp(T / 2, BANDWIDTH, N // 2, smoothing=5).with_peak(peak),
        nt.delay(T / 2, N // 2),
    ]
    cycle = nt.PhaseCycle(
        np.column_stack([np.zeros(4), 90.0 * np.arange(4), np.zeros(4)]),
        90.0 * np.array([0, 2, 0, 2]),
    )
    offsets, scales = b1_axes(args, BANDWIDTH)
    finish_b1(
        "b1_double_chirp",
        args,
        sequence,
        offsets,
        scales,
        rf_ref=peak,
        sign=1,
        phase_cycle=cycle,
    )


if __name__ == "__main__":
    main()
