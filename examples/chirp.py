"""Linear chirp excitation.

A 10 ms chirp sweeping 20 kHz in 500 steps, 10 % smoothing, Rabi frequency set for a 90°
flip; 31 offsets across ±5 kHz.
"""

import nutate as nt
from nutate.pulses.swept_frequency import chirp


def main() -> None:
    pulse = chirp(10e-3, 20e3, 500, smoothing=10)
    result = nt.run(
        pulse, 10e3, name="chirp", still=True, out="output", video_options={"duration": 12}
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
