"""CHORUS broadband excitation with phase correction (NMR).

t90 = 2 ms, t180 = 1 ms, 1 µs steps, 300 kHz, 5 % smoothing, degree-10 phase correction.
The video shows 31 offsets across ±15 kHz; excitation profiles over ±150 kHz (16-step
phase cycle) are saved before and after the correction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nutate as nt
from nutate.pulses.swept_frequency import CHORUS_CYCLE, chorus, chorus_phase_correction

T90, T180, BANDWIDTH, DT = 2e-3, 1e-3, 300e3, 1e-6


def main() -> None:
    correction = chorus_phase_correction(T90, T180, BANDWIDTH, DT, degree=10)
    band = np.linspace(-BANDWIDTH / 2, BANDWIDTH / 2, 101)
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    for label, corr in (("before", None), ("after", correction)):
        sequence = chorus(T90, T180, BANDWIDTH, DT, phase_correction_deg=corr)
        profile = nt.simulate(sequence, band, phase_cycle=CHORUS_CYCLE)
        fig = nt.plot_profiles(profile)
        fig.savefig(out / f"chorus_profiles_{label}.png", dpi=150)
        plt.close(fig)

    sequence = chorus(T90, T180, BANDWIDTH, DT, phase_correction_deg=correction)
    # 6000 steps: a short trail, or 31 offsets × many precession turns become a web.
    result = nt.run(
        sequence,
        30e3,
        name="chorus",
        still=True,
        out="output",
        video_options={"duration": 15, "trail": 0.02},
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
