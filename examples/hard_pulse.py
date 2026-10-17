"""Hard (rectangular) 90° pulse across a wide offset range.

Illustrative parameters: 10 µs 90° pulse in 1000 steps, 5 offsets across ±50 kHz.
"""

import numpy as np
from _common import finish, parse_args

import nutate as nt


def main() -> None:
    args = parse_args(__doc__)
    traj = nt.simulate([nt.hard(10e-6, n_steps=1000)], np.linspace(-50e3, 50e3, 5))
    finish(traj, "hard_pulse", args)


if __name__ == "__main__":
    main()
