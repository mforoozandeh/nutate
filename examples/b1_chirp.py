"""B1 robustness of a single chirp excitation.

A Q = 5 chirp with 5 % smoothing whose Rabi frequency is divided by 3.3588 (bringing it to a
90° chirp); offsets normalised by the unscaled Q = 5 Rabi frequency. Writes the −My map, its
surface and the on-resonance phase sensitivity. Illustrative parameters: 1 ms, 300 kHz,
1000 steps.
"""

import nutate as nt
from nutate.pulses.swept_frequency import chirp

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    pulse = chirp(T, BANDWIDTH, N, smoothing=5, q=5)
    result = nt.run(
        pulse.with_rabi(pulse.rabi / 3.3588),
        BANDWIDTH,
        name="b1_chirp",
        video=False,
        b1=True,
        rf_ref=pulse.rabi,
        # 1000 offsets as in the original script: ~120 phase cycles across the band alias at 101.
        b1_offsets=1000,
        out="output",
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
