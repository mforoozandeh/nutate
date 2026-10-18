"""Composite pulses.

t90 = 10 µs (Ω1 = 25 kHz); 31 offsets across ±window·Ω1 (--window, default 0.8). A
composite's behaviour depends only on Δ/Ω1, so keep the window within the range the preset
was designed for. Choose a preset with --preset.
"""

from _common import parse_args, report, run_options

import nutate as nt

T90 = 10e-6  # Ω1 = 1/(4·t90) = 25 kHz


def _options(parser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(nt.COMPOSITE_PRESETS, key=int),
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
    flips, phases = nt.COMPOSITE_PRESETS[args.preset]
    result = nt.run(
        nt.composite(T90, flips, phases),
        2 * args.window / (4 * T90),  # ±window·Ω1
        name="composite",
        still=True,
        **run_options(args, inset_phase=True),
    )
    report(result)


if __name__ == "__main__":
    main()
