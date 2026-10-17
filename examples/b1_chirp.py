"""B1 robustness of a single chirp excitation.

A Q = 5 chirp with 5 % smoothing whose amplitude is divided by 3.3588 (bringing it to a 90°
chirp); offsets normalised by the unscaled Q = 5 peak. Writes the −My map, its surface and
the on-resonance phase sensitivity. Illustrative parameters: 1 ms, 300 kHz, 1000 steps.
"""

import numpy as np
from _common import b1_axes, finish_b1, parse_args

import nutate as nt

T, BANDWIDTH, N = 1e-3, 300e3, 1000


def main() -> None:
    args = parse_args(__doc__)
    peak = np.sqrt(T * BANDWIDTH * 5 / (2 * np.pi)) / T
    pulse = nt.chirp(T, BANDWIDTH, N, smoothing=5, q=5).with_peak(peak / 3.3588)
    # noffs = 1000 as in the script: ~120 phase cycles across the band alias at 101.
    offsets, scales = b1_axes(args, BANDWIDTH, n_offsets=1000)
    finish_b1("b1_chirp", args, [pulse], offsets, scales, rf_ref=peak, sign=-1)


if __name__ == "__main__":
    main()
