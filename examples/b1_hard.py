"""B1 robustness of a hard 90° pulse.

A single rectangular step, 4-step cycle (pulse and receiver phase 0/90/180/270°). Writes the
−My map over Δ/Ω1(0) × Ω1/Ω1(0), its surface, and the on-resonance phase sensitivity.
Illustrative parameters: 10 µs pulse, offsets across ±100 kHz (±4 Ω1).
"""

import numpy as np
from _common import parse_args, report, run_options

import nutate as nt

T, BANDWIDTH = 10e-6, 200e3


def main() -> None:
    args = parse_args(__doc__)
    cycle = 90.0 * np.arange(4)
    result = nt.run(
        nt.hard(T, n_steps=1),
        BANDWIDTH,
        name="b1_hard",
        video=False,
        b1=True,
        phase_cycle=nt.PhaseCycle(cycle[:, None], cycle),
        **run_options(args),
    )
    report(result)


if __name__ == "__main__":
    main()
