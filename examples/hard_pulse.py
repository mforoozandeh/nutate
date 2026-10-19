"""Hard (rectangular) 90° pulse across a wide offset range.

The waveform is just ones: ``nt.waveform`` with ``t90`` equal to its duration gives the Rabi
frequency 1/(4·t90) of a 90° pulse. Illustrative parameters: 10 µs pulse in 1000 steps,
5 offsets across ±50 kHz.
"""

import numpy as np

import nutate as nt

T90, N = 10e-6, 1000


def main() -> None:
    pulse = nt.waveform(np.ones(N), np.zeros(N), duration=T90, t90=T90, name="hard 90°")
    result = nt.run(pulse, 100e3, name="hard_pulse", n_offsets=5, still=True, out="output")
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
