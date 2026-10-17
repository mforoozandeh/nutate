"""Chirp excitation followed by an ideal rewind of its quadratic phase.

The 10 ms, 20 kHz chirp of chirp.py, then 500 steps of free precession at the remapped
offset Δ·(Δ/(2·bw) − 1/2), which refocuses the chirp's offset-dependent phase. 31 offsets
across ±2 kHz.
"""

import numpy as np
from _common import finish, parse_args

import nutate as nt

BANDWIDTH = 20e3


def main() -> None:
    args = parse_args(__doc__)
    sequence = [
        nt.chirp(10e-3, BANDWIDTH, 500, smoothing=10),
        nt.delay(10e-3, 500, name="rewind", offset_fn=lambda d: d * (d / (2 * BANDWIDTH) - 0.5)),
    ]
    traj = nt.simulate(sequence, np.linspace(-2e3, 2e3, 31))
    finish(traj, "chirp_rewind", args, duration=14)


if __name__ == "__main__":
    main()
