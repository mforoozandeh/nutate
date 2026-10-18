"""B1 robustness of CHORUS (equal-duration variant).

Chirps of T, T and T/2 with a T/2 delay before the last, at 1/4.75, 0.9661/√2 and 1 × the
Q = 5 Rabi frequency for T/2, 5 % smoothing, degree-10 phase correction of the first chirp,
16-step cycle. Writes the −My map, its surface and the on-resonance phase sensitivity.
Illustrative parameters: T = 1 ms, 300 kHz, 1000 steps.
"""

import numpy as np
from _common import parse_args, report, run_options

import nutate as nt

T, BANDWIDTH, N = 1e-3, 300e3, 1000
RABI = np.sqrt(T / 2 * BANDWIDTH * 5 / (2 * np.pi)) / (T / 2)


def sequence(correction=None) -> list[nt.Segment]:
    return [
        nt.chirp(T, BANDWIDTH, N, smoothing=5, phase_correction_deg=correction).with_rabi(
            RABI / 4.75
        ),
        nt.chirp(T, BANDWIDTH, N, smoothing=5).with_rabi(0.9661 * RABI / np.sqrt(2)),
        nt.delay(T / 2, N // 2),
        nt.chirp(T / 2, BANDWIDTH, N // 2, smoothing=5).with_rabi(RABI),
    ]


def main() -> None:
    args = parse_args(__doc__)
    correction = nt.phase_correction(
        sequence(), BANDWIDTH, phase_cycle=nt.CHORUS_CYCLE, degree=10, n_offsets=100
    )
    result = nt.run(
        sequence(correction),
        BANDWIDTH,
        name="b1_chorus",
        video=False,
        b1=True,
        phase_cycle=nt.CHORUS_CYCLE,
        **run_options(args),
    )
    report(result)


if __name__ == "__main__":
    main()
