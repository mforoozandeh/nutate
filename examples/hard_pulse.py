"""Hard (rectangular) 90° pulse across a wide offset range.

Illustrative parameters: 10 µs 90° pulse in 1000 steps, 5 offsets across ±50 kHz.
"""

import nutate as nt


def main() -> None:
    pulse = nt.hard(10e-6, n_steps=1000)
    result = nt.run(pulse, 100e3, name="hard_pulse", n_offsets=5, still=True, out="output")
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
