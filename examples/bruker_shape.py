"""Simulate a pulse stored as a Bruker (JCAMP-DX) shape file.

Example: a 2.5 ms OCTAPUS shape at a 15.1388 kHz Rabi frequency (100 %) and 360 kHz
bandwidth, simulated at 5 offsets across the band:

    python examples/bruker_shape.py OCTPul...txt 2.5e-3 15138.8 --bandwidth 360e3
"""

from _common import parse_args, report, run_options

import nutate as nt


def _options(parser) -> None:
    parser.add_argument("path", help="Bruker shape file")
    parser.add_argument("duration", type=float, help="pulse duration, s")
    parser.add_argument("rabi", type=float, help="Rabi frequency at 100 %%, Hz")
    parser.add_argument(
        "--bandwidth", type=float, default=360e3, help="offset range, Hz (default: 360e3)"
    )


def main() -> None:
    args = parse_args(__doc__, _options)
    pulse = nt.read_bruker_shape(args.path, args.duration, args.rabi)
    result = nt.run(
        pulse, args.bandwidth, name="bruker_shape", n_offsets=5, still=True, **run_options(args)
    )
    report(result)


if __name__ == "__main__":
    main()
