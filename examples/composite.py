"""Composite pulse.

Preset 9 of nutate.pulses.composite.PRESETS (119°, 183°, 211°, 384°, 211°, 183°, 119° at
alternating phases) built on a 10 µs 90° pulse, which sets the Rabi frequency
1/(4·t90) = 25 kHz; 31 offsets across ±0.8·ν1.
"""

import nutate as nt
from nutate.pulses.composite import PRESETS, composite

T90 = 10e-6


def main() -> None:
    pulse = composite(T90, *PRESETS["9"])
    rabi = 1 / (4 * T90)
    result = nt.run(
        pulse,
        1.6 * rabi,
        name="composite",
        still=True,
        out="output",
        video_options={"inset_phase": True, "theme": "dark"},
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
