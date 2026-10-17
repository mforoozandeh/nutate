"""B1 robustness of a hard 90° pulse.

A single rectangular step, 4-step cycle (pulse and receiver phase 0/90/180/270°). Writes the
−My map over Δ/Ω1(0) × Ω1/Ω1(0), its surface, and the on-resonance phase sensitivity.
Illustrative parameters: 10 µs pulse, offsets across ±100 kHz (±4 Ω1).
"""

import numpy as np
from _common import b1_axes, finish_b1, parse_args

import nutate as nt

T, BANDWIDTH = 10e-6, 200e3


def main() -> None:
    args = parse_args(__doc__)
    cycle = 90.0 * np.arange(4)
    offsets, scales = b1_axes(args, BANDWIDTH)
    finish_b1(
        "b1_hard",
        args,
        [nt.hard(T, n_steps=1)],
        offsets,
        scales,
        rf_ref=1 / (4 * T),
        sign=-1,
        phase_cycle=nt.PhaseCycle(cycle[:, None], cycle),
    )


if __name__ == "__main__":
    main()
