"""Hard (rectangular) 90° pulse across a wide offset range.

Illustrative parameters: 10 µs 90° pulse in 1000 steps, 5 offsets across ±50 kHz.
"""

from _common import parse_args, report, run_options

import nutate as nt


def main() -> None:
    args = parse_args(__doc__)
    pulse = nt.hard(10e-6, n_steps=1000)
    report(nt.run(pulse, 100e3, name="hard_pulse", n_offsets=5, still=True, **run_options(args)))


if __name__ == "__main__":
    main()
