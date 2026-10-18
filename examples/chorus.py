"""CHORUS broadband excitation with phase correction (NMR).

t90 = 2 ms, t180 = 1 ms, 1 µs steps, 300 kHz, 5 % smoothing, degree-10 phase correction.
The video shows 31 offsets across ±15 kHz; excitation profiles over ±150 kHz (16-step
phase cycle) are saved before and after the correction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, report, run_options

import nutate as nt

T90, T180, BANDWIDTH, DT = 2e-3, 1e-3, 300e3, 1e-6


def main() -> None:
    args = parse_args(__doc__)
    correction = nt.chorus_phase_correction(T90, T180, BANDWIDTH, DT, degree=10)
    band = np.linspace(-BANDWIDTH / 2, BANDWIDTH / 2, 101)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for label, corr in (("before", None), ("after", correction)):
        sequence = nt.chorus(T90, T180, BANDWIDTH, DT, phase_correction_deg=corr)
        profile = nt.simulate(sequence, band, phase_cycle=nt.CHORUS_CYCLE)
        fig = nt.plot_profiles(profile)
        fig.savefig(out / f"chorus_profiles_{label}.png", dpi=150)
        plt.close(fig)

    sequence = nt.chorus(T90, T180, BANDWIDTH, DT, phase_correction_deg=correction)
    # 6000 steps: a short trail, or 31 offsets × many precession turns become a web.
    options = run_options(args, duration=15, trail=0.02)
    report(nt.run(sequence, 30e3, name="chorus", still=True, **options))


if __name__ == "__main__":
    main()
