"""Compressed CHORUS for EPR (ns pulses, MHz bandwidths).

Illustrative parameters: t90 = 200 ns, t180 = 100 ns, 200 MHz, 0.625 ns steps (Bruker AWG
resolution), 5 % smoothing, degree-4 phase correction.

Outputs: the video (5 offsets from −0.1·bw to +0.3·bw), a map of −My over time and offset
(101 offsets across ±0.3·bw, with a trailing t90 delay), and phase paths atan2(My, Mx) for
3 offsets across ±0.05·bw with all pulse phases at 90° (receiver phase not applied).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nutate as nt

T90, T180, BANDWIDTH, DT = 200e-9, 100e-9, 200e6, 0.625e-9
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: distinct for colour-blind readers


def phase_paths(sequence: list[nt.Segment], offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times (s) and atan2(My, Mx)/π per step, shape (steps + 1, offsets).

    Every pulse phase is advanced by 90°; the receiver phase is not applied to the path.
    """
    phased = nt.PhaseCycle(np.full((1, len(sequence)), 90.0), [0.0])
    traj = nt.simulate(sequence, offsets, phase_cycle=phased)
    return traj.t, np.arctan2(traj.m[:, :, 1], traj.m[:, :, 0]) / np.pi


def main() -> None:
    correction = nt.chorus_phase_correction(T90, T180, BANDWIDTH, DT, degree=4)
    sequence = nt.chorus(T90, T180, BANDWIDTH, DT, phase_correction_deg=correction)
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    with_delay = [*sequence, nt.delay(T90, sequence[0].n_steps)]
    mapped = nt.simulate(with_delay, np.linspace(-0.3 * BANDWIDTH, 0.3 * BANDWIDTH, 101))
    fig = nt.plot_map(mapped, "y", sign=-1)
    fig.savefig(out / "compressed_chorus_map.png", dpi=150)
    plt.close(fig)

    path_offsets = np.linspace(-0.05 * BANDWIDTH, 0.05 * BANDWIDTH, 3)
    t, paths = phase_paths(sequence, path_offsets)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    for j, (offset, colour) in enumerate(zip(path_offsets, COLOURS)):
        ax.plot(t * 1e9, paths[:, j], color=colour, linewidth=1.5, label=f"{offset / 1e6:g} MHz")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("atan2(My, Mx) / π")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(out / "compressed_chorus_phase_paths.png", dpi=150)
    plt.close(fig)

    result = nt.run(
        sequence,
        offsets=np.linspace(-0.1 * BANDWIDTH, 0.3 * BANDWIDTH, 5),
        name="compressed_chorus_epr",
        still=True,
        out="output",
        video_options={"duration": 12},
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
