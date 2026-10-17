"""Linear chirp excitation.

A 10 ms chirp sweeping 20 kHz in 500 steps, 10 % smoothing, amplitude set for a 90° flip;
31 offsets across ±5 kHz.
"""

import numpy as np
from _common import finish, parse_args

import nutate as nt


def main() -> None:
    args = parse_args(__doc__)
    pulse = nt.chirp(10e-3, 20e3, 500, smoothing=10)
    traj = nt.simulate([pulse], np.linspace(-5e3, 5e3, 31))
    finish(traj, "chirp", args, duration=12)


if __name__ == "__main__":
    main()
