"""B1 robustness of a hard 90° pulse.

A single rectangular step, 4-step cycle (pulse and receiver phase 0/90/180/270°). Writes the
−My map over Δ/Ω1(0) × Ω1/Ω1(0), its surface, and the on-resonance phase sensitivity.
Illustrative parameters: 10 µs pulse, offsets across ±100 kHz (±4 Ω1).
"""

import numpy as np

import nutate as nt

T, BANDWIDTH = 10e-6, 200e3


def main() -> None:
    cycle = 90.0 * np.arange(4)
    result = nt.run(
        nt.waveform([1.0], [0.0], duration=T, t90=T, name="hard 90°"),  # one step
        BANDWIDTH,
        name="b1_hard",
        video=False,
        b1=True,
        phase_cycle=nt.PhaseCycle(cycle[:, None], cycle),
        out="output",
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
