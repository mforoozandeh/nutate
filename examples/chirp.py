"""Linear chirp excitation.

A 10 ms chirp sweeping 20 kHz in 500 steps, 10 % smoothing, Rabi frequency set for a 90°
flip; 31 offsets across ±5 kHz.
"""

from _common import parse_args, report, run_options

import nutate as nt


def main() -> None:
    args = parse_args(__doc__)
    pulse = nt.chirp(10e-3, 20e3, 500, smoothing=10)
    report(nt.run(pulse, 10e3, name="chirp", still=True, **run_options(args, duration=12)))


if __name__ == "__main__":
    main()
