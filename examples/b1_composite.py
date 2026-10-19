"""B1 robustness of a composite pulse.

Preset 9 of nutate.pulses.composite.PRESETS built on a 10 µs 90° pulse, which sets the Rabi
frequency 1/(4·t90) = 25 kHz; offsets across ±0.8·ν1. Writes the −My map, its surface and
the on-resonance phase sensitivity.
"""

import nutate as nt
from nutate.pulses.composite import PRESETS, composite

T90 = 10e-6


def main() -> None:
    pulse = composite(T90, *PRESETS["9"])
    rabi = 1 / (4 * T90)
    result = nt.run(pulse, 1.6 * rabi, name="b1_composite", video=False, b1=True, out="output")
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
