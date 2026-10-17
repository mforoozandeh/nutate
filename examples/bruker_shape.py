"""Simulate a pulse stored as a Bruker (JCAMP-DX) shape file.

Example: a 2.5 ms OCTAPUS shape with 15.1388 kHz peak RF and 360 kHz bandwidth, simulated
at 5 offsets across the band:

    python examples/bruker_shape.py OCTPul...txt 2.5e-3 15138.8 --bandwidth 360e3
"""

import numpy as np
from _common import finish, parse_args

import nutate as nt


def _options(parser) -> None:
    parser.add_argument("path", help="Bruker shape file")
    parser.add_argument("duration", type=float, help="pulse duration, s")
    parser.add_argument("rf_max", type=float, help="RF amplitude at 100 %%, Hz")
    parser.add_argument(
        "--bandwidth", type=float, default=360e3, help="offset range, Hz (default: 360e3)"
    )


def main() -> None:
    args = parse_args(__doc__, _options)
    pulse = nt.read_bruker_shape(args.path, args.duration, args.rf_max)
    offsets = np.linspace(-args.bandwidth / 2, args.bandwidth / 2, 5)
    finish(nt.simulate([pulse], offsets), "bruker_shape", args)


if __name__ == "__main__":
    main()
