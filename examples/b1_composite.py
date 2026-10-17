"""B1 robustness of composite pulses.

This example's preset table differs from nutate.COMPOSITE_PRESETS in 10 (same as 4), 12
(phases 90/0/90) and 14 (a 48-element train of 90° pulses). Offsets span ±window·Ω1
(--window, default 0.8). Writes the −My map, its surface and the on-resonance phase
sensitivity. Illustrative parameters: t90 = 10 µs.
"""

from _common import b1_axes, finish_b1, parse_args

import nutate as nt

T90 = 10e-6

# (flips_deg, phases_deg) presets for this example.
PRESET_14_PHASES = """
    0 90 180 180 270 180 0 90 0 0 270 180 0 90 0 0 270 180 0 90
    180 180 270 180 0 270 180 180 90 180 0 270 0 0 90 180 0 270 0 0 90 180 0 270 180 180 90 180
"""
PRESETS = {
    "1": ((90,), (0,)),
    "2": ((90, 90), (0, 90)),
    "3": ((90, 180), (0, 120)),
    "4": ((180, 360, 180, 90), (97.2, 291.5, 97.2, 0)),
    "5": ((180, 360, 180, 270, 90), (0, 180, 0, 180, 90)),
    "6": ((385, 320, 25), (0, 180, 0)),
    "7": ((113, 316, 113), (180, 0, 180)),
    "8": ((24, 152, 346, 152, 24), (0, 180, 0, 180, 0)),
    "9": ((119, 183, 211, 384, 211, 183, 119), (180, 0, 180, 0, 180, 0, 180)),
    "10": ((180, 360, 180, 90), (97.2, 291.5, 97.2, 0)),
    "11": ((180, 360, 180, 270, 90, 360, 270, 90), (0, 180, 0, 180, 90, 0, 180, 90)),
    "12": ((90, 180, 90), (90, 0, 90)),
    "13": ((180,), (0,)),
    "14": ((90,) * 48, tuple(float(v) for v in PRESET_14_PHASES.split())),
}


def _options(parser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS, key=int),
        default="9",
        help="composite pulse preset (default: 9)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.8,
        help="offsets span ±window·Ω1 (default: 0.8)",
    )


def main() -> None:
    args = parse_args(__doc__, _options)
    flips, phases = PRESETS[args.preset]
    offsets, scales = b1_axes(args, 2 * args.window / (4 * T90))
    sequence = nt.composite(T90, flips, phases)
    finish_b1("b1_composite", args, sequence, offsets, scales, rf_ref=1 / (4 * T90), sign=-1)


if __name__ == "__main__":
    main()
